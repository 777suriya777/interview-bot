"""
Sentiment Service — FastAPI inference endpoint for confidence classification.

Endpoints:
  POST /classify  → label a transcript as high/moderate/low/anxious
  GET  /health    → liveness probe

Key implementation details:
  1. Model + vocab loaded ONCE at startup via lifespan
  2. If model weights are absent (pre-training), service still starts —
     returns a deterministic rule-based fallback using hedging count
  3. confidence_score = softmax probability of the predicted class
  4. hedging_phrases and filler_word_count come from analyzer.py (torch-free)
  5. No Redis cache here — transcripts vary too much for cache hits to be useful
     (unlike NLP which caches identical question+answer pairs)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analyzer import (
    build_confidence_feedback,
    count_filler_words,
    detect_hedging,
    IDX_TO_LABEL,
)
from model import (
    ConfidenceLSTM,
    encode_batch,
    load_vocab,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)

# Shared state populated at startup
model_store: dict = {}


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and vocab once at startup; clean up on shutdown."""
    logger.info("Sentiment service starting up…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    model_path = os.environ.get("LSTM_MODEL_PATH", "/models/confidence_lstm.pt")
    vocab_path  = os.environ.get("VOCAB_PATH",      "/models/vocab.json")

    model: Optional[ConfidenceLSTM] = None
    vocab: Optional[dict] = None

    # ── Load vocabulary ────────────────────────────────────────────────
    if os.path.exists(vocab_path):
        try:
            vocab = load_vocab(vocab_path)
            logger.info(f"✓ Vocabulary loaded: {len(vocab)} tokens")
        except Exception as e:
            logger.warning(f"Vocab load failed: {e}. Falling back to rule-based classifier.")
    else:
        logger.warning(
            f"Vocab file not found at {vocab_path}. "
            "Run sentiment_service/train.py to generate vocab.json."
        )

    # ── Load model weights ─────────────────────────────────────────────
    if vocab and os.path.exists(model_path):
        try:
            model = ConfidenceLSTM(vocab_size=len(vocab))
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info("✓ ConfidenceLSTM weights loaded")
        except Exception as e:
            logger.warning(f"Model load failed: {e}. Falling back to rule-based classifier.")
            model = None
    else:
        logger.warning(
            f"Model file not found at {model_path}. "
            "Starting with rule-based fallback classifier."
        )

    model_store["model"]  = model   # None = use fallback
    model_store["vocab"]  = vocab
    model_store["device"] = device
    model_store["ready"]  = True

    logger.info("Sentiment service ready.")
    yield

    logger.info("Sentiment service shutting down…")
    model_store.clear()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Bot — Sentiment Service",
    description="BiLSTM confidence classifier for interview answer transcripts",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Candidate's answer transcript")


class HedgingPhrase(BaseModel):
    text:       str
    start_char: int


class ClassifyResponse(BaseModel):
    confidence_label:  str   = Field(..., description="high|moderate|low|anxious")
    confidence_score:  float = Field(..., ge=0.0, le=1.0,
                                     description="Softmax probability of predicted class")
    hedging_phrases:   list[HedgingPhrase]
    filler_word_count: int
    feedback_text:     str


class HealthResponse(BaseModel):
    status: str
    model:  str
    loaded: bool
    device: str


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_classify(transcript: str) -> tuple[str, float]:
    """
    Fallback classifier used when no trained model is available.
    Uses hedging phrase count per 100 words — matches the annotation
    guideline from the spec: '>3 hedges per 100 words → Low or Anxious'.

    Returns (label, confidence_score).
    """
    hedges = detect_hedging(transcript)
    fillers = count_filler_words(transcript)
    word_count = max(len(transcript.split()), 1)

    hedges_per_100 = (len(hedges) / max(word_count, 50)) * 100
    fillers_per_100 = (fillers / max(word_count, 50)) * 100

    if hedges_per_100 > 6 or fillers_per_100 > 8:
        return "anxious", 0.65
    elif hedges_per_100 > 3:
        return "low", 0.60
    elif hedges_per_100 > 1:
        return "moderate", 0.65
    else:
        return "high", 0.70


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(transcript: str) -> tuple[str, float]:
    """
    Run the BiLSTM model on a single transcript.
    Returns (label, softmax_probability).
    """
    model: Optional[ConfidenceLSTM] = model_store.get("model")
    vocab: Optional[dict]           = model_store.get("vocab")
    device: torch.device            = model_store.get("device", torch.device("cpu"))

    # Fall back to rule-based if model not loaded
    if model is None or vocab is None:
        return _rule_based_classify(transcript)

    # Encode: tokenise → pad → tensor of shape [1, seq_len]
    x = encode_batch([transcript], vocab, max_length=300).to(device)

    with torch.no_grad():
        logits = model(x)                      # [1, 4]
        probs  = F.softmax(logits, dim=-1)     # [1, 4]
        pred_idx = int(probs.argmax(dim=-1).item())
        pred_prob = float(probs[0, pred_idx].item())

    label = IDX_TO_LABEL.get(pred_idx, "moderate")
    return label, round(pred_prob, 4)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Classify the confidence level of a candidate's transcript.

    Returns the confidence label (high/moderate/low/anxious),
    softmax probability, detected hedging phrases, filler word count,
    and an actionable feedback string.
    """
    if not model_store.get("ready"):
        raise HTTPException(status_code=503, detail="Service not ready")

    # ── Lexical analysis (always run, torch-free) ─────────────────────
    hedging_phrases = detect_hedging(req.transcript)
    filler_count    = count_filler_words(req.transcript)

    # ── Model inference ───────────────────────────────────────────────
    try:
        label, score = _run_inference(req.transcript)
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        # Degrade gracefully — use rule-based fallback, don't crash the session
        label, score = _rule_based_classify(req.transcript)

    # ── Build feedback ────────────────────────────────────────────────
    feedback = build_confidence_feedback(
        label=label,
        hedging_count=len(hedging_phrases),
        filler_count=filler_count,
    )

    return ClassifyResponse(
        confidence_label  = label,
        confidence_score  = score,
        hedging_phrases   = [HedgingPhrase(**p) for p in hedging_phrases],
        filler_word_count = filler_count,
        feedback_text     = feedback,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe for Docker healthcheck."""
    model_loaded = model_store.get("model") is not None
    device_name  = str(model_store.get("device", "unknown"))

    return HealthResponse(
        status = "ok" if model_store.get("ready") else "loading",
        model  = "confidence-lstm-bilstm" if model_loaded else "rule-based-fallback",
        loaded = model_loaded,
        device = device_name,
    )
