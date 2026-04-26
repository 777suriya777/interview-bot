"""
NLP Service — FastAPI inference endpoint for BERT answer evaluation.

Endpoints:
  POST /evaluate   → score a question-answer pair (4 dims + overall + feedback)
  GET  /health     → liveness probe for Docker healthcheck

Key implementation details:
  1. Model loaded ONCE at startup via lifespan (not on each request)
  2. Redis cache keyed by SHA-256(question + answer) — 1h TTL
     Avoids redundant GPU/CPU inference for identical Q+A pairs
     (e.g. when a user re-submits after a network error)
  3. Fallback: if Redis is unavailable, inference still works (no crash)
  4. Timeout budget: entire /evaluate should complete in < 2s for text answers
"""
import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import google.generativeai as genai
from transformers import BertTokenizer

from model import InterviewBERTScorer
from feedback import build_feedback_summary

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)

# ── Shared state loaded at startup ────────────────────────────────────────────
# Stored in a dict so lifespan can populate it — avoids global reassignment.
model_store: dict = {}


# ── Lifespan: load model and Redis once at startup ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Everything before `yield` runs at startup; after `yield` runs at shutdown.
    Using lifespan instead of deprecated @app.on_event("startup").
    """
    logger.info("NLP service starting up…")

    # ── Determine device ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    # ── Load tokeniser ────────────────────────────────────────────────
    logger.info("Loading BERT tokeniser (bert-base-uncased)…")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # ── Load fine-tuned model weights ─────────────────────────────────
    model_path = os.environ.get("BERT_MODEL_PATH", "/models/bert_scorer.pt")
    logger.info(f"Loading model weights from: {model_path}")

    model = InterviewBERTScorer()

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✓ Fine-tuned weights loaded successfully")
        model_store["is_trained"] = True
    else:
        # No trained weights yet — model uses random BERT fine-tuning weights.
        # This allows the service to start during development before training.
        logger.warning(
            f"Model file not found at {model_path}. "
            "Starting with uninitialised classification heads. "
            "Run nlp_service/train.py to generate bert_scorer.pt."
        )
        model_store["is_trained"] = False

    model.to(device)
    model.eval()  # disable dropout for inference

    # ── Configure Gemini API client ────────────────────────────────────
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        model_store["gemini_model"] = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✓ Gemini API client configured (gemini-1.5-flash)")
    else:
        logger.warning("GEMINI_API_KEY not set — /generate endpoint will be disabled.")
        model_store["gemini_model"] = None

    # ── Connect to Redis (optional — service still works without it) ──
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    redis_client: Optional[aioredis.Redis] = None
    try:
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info(f"✓ Redis connected: {redis_url}")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}). Cache disabled — inference will still work.")
        redis_client = None

    # Populate shared state
    model_store["model"]     = model
    model_store["tokenizer"] = tokenizer
    model_store["device"]    = device
    model_store["redis"]     = redis_client

    logger.info("NLP service ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("NLP service shutting down…")
    if redis_client:
        await redis_client.aclose()
    model_store.clear()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Bot — NLP Service",
    description="BERT-based interview answer evaluation (4-dimensional scoring)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The interview question text")
    answer:   str = Field(..., min_length=1, description="The candidate's answer text")


class EvalResponse(BaseModel):
    content_score:      int   = Field(..., ge=0, le=4)
    relevance_score:    int   = Field(..., ge=0, le=4)
    completeness_score: int   = Field(..., ge=0, le=4)
    accuracy_score:     int   = Field(..., ge=0, le=4)
    overall_score:      float = Field(..., ge=0.0, le=4.0)
    feedback_summary:   str


class HealthResponse(BaseModel):
    status: str
    model:  str
    loaded: bool
    device: str
    cache:  str


class GenerateRequest(BaseModel):
    resume_text: str = Field(..., description="Parsed resume text to generate a question from")

class GenerateResponse(BaseModel):
    question: str


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(question: str, answer: str) -> str:
    """
    SHA-256 hash of the concatenated question+answer strings.
    Spec: NLP cache key = SHA-256(question + answer), prefix nlp:cache:
    Identical Q+A pairs hash to the same key → no redundant GPU inference.
    """
    content = (question + answer).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    return f"nlp:cache:{digest}"


async def _get_cached(question: str, answer: str) -> Optional[dict]:
    """Return cached evaluation result or None if cache miss / Redis down."""
    redis: Optional[aioredis.Redis] = model_store.get("redis")
    if not redis:
        return None
    try:
        key = _cache_key(question, answer)
        raw = await redis.get(key)
        if raw:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Redis GET failed: {e}")
    return None


async def _set_cached(question: str, answer: str, result: dict) -> None:
    """Store evaluation result in Redis with 1h TTL. Silently fails if Redis is down."""
    redis: Optional[aioredis.Redis] = model_store.get("redis")
    if not redis:
        return
    try:
        key = _cache_key(question, answer)
        # TTL = 3600s (1 hour) per spec
        await redis.setex(key, 3600, json.dumps(result))
        logger.debug(f"Cache SET: {key}")
    except Exception as e:
        logger.warning(f"Redis SET failed: {e}")


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(question: str, answer: str) -> dict:
    """
    Tokenise the Q+A pair and run forward pass through the BERT model.
    Returns raw score dict before building the full response.
    Runs synchronously — called from the async endpoint via the event loop
    (acceptable since inference is CPU-bound and fast; for GPU use
    asyncio.to_thread() if inference time > 100ms).
    """
    if not model_store.get("is_trained", False):
        # Fallback rule-based scorer (as specified in README Option A)
        q_words = set(question.lower().replace('?', ' ').replace('.', ' ').split())
        a_words = set(answer.lower().replace('.', ' ').split())
        overlap = len(q_words.intersection(a_words))
        
        # Simple heuristic
        length_score = min(4, len(a_words) // 15)
        rel_score = min(4, overlap // 2) if overlap > 0 else 0
        
        return {
            "content": max(1, length_score),
            "relevance": max(1, rel_score + 1),
            "completeness": max(1, (length_score + rel_score) // 2),
            "accuracy": max(1, rel_score + 1)
        }

    tokenizer = model_store["tokenizer"]
    model     = model_store["model"]
    device    = model_store["device"]

    # Tokenise as a sentence pair: [CLS] question [SEP] answer [SEP]
    enc = tokenizer(
        question,
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,  # no padding needed for single-sample inference
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
        )

    # Convert logits to integer scores via argmax
    scores = {
        dim: int(logits[dim].argmax(dim=-1).item())
        for dim in InterviewBERTScorer.DIMENSIONS
    }
    return scores


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest) -> EvalResponse:
    """
    Score a candidate's answer to an interview question.

    Returns 4 dimension scores (0-4 each) + weighted overall + feedback text.
    Uses Redis cache to avoid re-inferencing identical Q+A pairs.
    """
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    # ── Cache check ───────────────────────────────────────────────────
    cached = await _get_cached(req.question, req.answer)
    if cached:
        return EvalResponse(**cached)

    # ── Inference ─────────────────────────────────────────────────────
    try:
        scores = _run_inference(req.question, req.answer)
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    # ── Build response ────────────────────────────────────────────────
    overall = round((
        scores["content"] + scores["relevance"] +
        scores["completeness"] + scores["accuracy"]
    ) / 4.0, 2)

    feedback = build_feedback_summary(scores)

    result = {
        "content_score":      scores["content"],
        "relevance_score":    scores["relevance"],
        "completeness_score": scores["completeness"],
        "accuracy_score":     scores["accuracy"],
        "overall_score":      overall,
        "feedback_summary":   feedback,
    }

    # ── Store in cache ────────────────────────────────────────────────
    await _set_cached(req.question, req.answer, result)

    return EvalResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness probe.
    Returns 200 with loaded=True once the BERT model is ready.
    Docker healthcheck polls this endpoint.
    """
    loaded = "model" in model_store
    redis_status = "connected" if model_store.get("redis") else "unavailable"
    device_name = str(model_store.get("device", "unknown"))

    return HealthResponse(
        status="ok" if loaded else "loading",
        model="bert-base-uncased",
        loaded=loaded,
        device=device_name,
        cache=redis_status,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_question(req: GenerateRequest) -> GenerateResponse:
    """
    Generate a dynamic technical interview question based on the resume.
    Uses Google Gemini API (gemini-1.5-flash) — no local model required.
    """
    gemini_model = model_store.get("gemini_model")
    if not gemini_model:
        raise HTTPException(
            status_code=503,
            detail="Gemini API not configured. Set GEMINI_API_KEY environment variable."
        )

    # Truncate resume to first 2000 chars to stay well within token limits
    snippet = req.resume_text[:2000]
    prompt = (
        "You are a senior technical interviewer. Based on the candidate's resume below, "
        "generate exactly ONE specific, challenging technical interview question that tests "
        "a key skill mentioned in their resume. Return only the question itself, with no "
        "preamble, numbering, or explanation.\n\n"
        f"Resume:\n{snippet}\n\n"
        "Interview Question:"
    )

    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        question_text = response.text.strip()
        if not question_text:
            raise ValueError("Empty response from Gemini")
        return GenerateResponse(question=question_text)
    except Exception as e:
        logger.error(f"[Gemini] Question generation failed: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")
