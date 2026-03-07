"""
Adaptive Engine — FastAPI service for intelligent question selection.

Endpoints:
  POST /next-question              → select next question for a session
  GET  /performance/:user_id       → get user's topic performance summary
  GET  /health                     → liveness probe

Data flow:
  1. /next-question receives last_scores + session context identifiers
  2. Load session state from Redis (or create fresh if first question)
  3. Load question bank from Redis cache (or DB if cache miss)
  4. Load user performance from Redis cache (or DB if cache miss)
  5. Run selector.select_next_question()
  6. Update session state in Redis
  7. Update user_performance table in DB + Redis cache
  8. Return SelectionResult

Redis keys (TTLs from spec):
  session:{session_id}:state     TTL 24h  — SessionContext JSON
  user:{user_id}:performance     TTL 1h   — per-topic score dict
  question_bank:active           TTL 10m  — list of QuestionRecord dicts

Error handling:
  - Redis unavailable → proceed without cache (slower but correct)
  - DB unavailable    → return fallback question + log error
  - No unseen questions → return random medium-difficulty from bank
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from selector import (
    QuestionRecord,
    SelectionResult,
    SessionContext,
    adjust_difficulty,
    select_fallback_question,
    select_next_question,
    update_performance_ema,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)

# ── Redis TTLs (spec Section 5.3) ─────────────────────────────────────────────
TTL_SESSION     = 86_400   # 24 hours
TTL_USER_PERF   = 3_600    # 1 hour
TTL_QUESTION_BANK = 600    # 10 minutes

# ── Shared state ──────────────────────────────────────────────────────────────
app_state: dict = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Adaptive engine starting up…")

    # PostgreSQL connection pool
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql://", 1)
    # asyncpg uses postgresql:// not postgresql+asyncpg://
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    pool = None
    try:
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        logger.info("✓ PostgreSQL pool created")
    except Exception as e:
        logger.error(f"DB pool creation failed: {e}. Service will use fallback mode.")

    # Redis connection
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    redis_client = None
    try:
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("✓ Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Running without cache.")
        redis_client = None

    app_state["pool"]   = pool
    app_state["redis"]  = redis_client
    app_state["ready"]  = True

    yield

    logger.info("Adaptive engine shutting down…")
    if pool:
        await pool.close()
    if redis_client:
        await redis_client.aclose()
    app_state.clear()


app = FastAPI(
    title="Interview Bot — Adaptive Engine",
    description="Intelligent question selection based on real-time performance",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LastScores(BaseModel):
    content:      int = Field(..., ge=0, le=4)
    relevance:    int = Field(..., ge=0, le=4)
    completeness: int = Field(..., ge=0, le=4)
    accuracy:     int = Field(..., ge=0, le=4)


class NextQuestionRequest(BaseModel):
    user_id:              str
    session_id:           str
    last_scores:          LastScores
    session_question_ids: list[str] = Field(default_factory=list)


class NextQuestionResponse(BaseModel):
    question_id:      str
    question_text:    str
    question_type:    str
    difficulty:       int
    user_level:       float
    selection_reason: str


class TopicStat(BaseModel):
    score:        float
    answer_count: int


class PerformanceResponse(BaseModel):
    overall_score:   float
    topics:          dict[str, TopicStat]
    priority_areas:  list[str]


class HealthResponse(BaseModel):
    status:     str
    db:         bool
    redis:      bool
    questions_cached: int


# ── Redis helpers ─────────────────────────────────────────────────────────────

async def _redis_get(key: str) -> Optional[str]:
    """Get a value from Redis, return None on any error."""
    r = app_state.get("redis")
    if not r:
        return None
    try:
        return await r.get(key)
    except Exception as e:
        logger.debug(f"Redis GET {key} failed: {e}")
        return None


async def _redis_set(key: str, value: str, ttl: int) -> None:
    """Set a value in Redis with TTL, silently ignore errors."""
    r = app_state.get("redis")
    if not r:
        return
    try:
        await r.setex(key, ttl, value)
    except Exception as e:
        logger.debug(f"Redis SET {key} failed: {e}")


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _load_questions_from_db() -> list[QuestionRecord]:
    """
    Fetch all active questions from PostgreSQL.
    Returns empty list on error (caller falls back to cache or raises).
    """
    pool = app_state.get("pool")
    if not pool:
        return []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text, text, type, difficulty, COALESCE(category, '') as category
                FROM questions
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                """
            )
        return [
            QuestionRecord(
                id         = row["id"],
                text       = row["text"],
                type       = row["type"],
                difficulty = row["difficulty"],
                category   = row["category"],
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"DB question fetch failed: {e}")
        return []


async def _load_user_performance_from_db(user_id: str) -> dict[str, float]:
    """
    Load per-topic scores from user_performance table.
    Returns empty dict on error.
    """
    pool = app_state.get("pool")
    if not pool:
        return {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT topic, score FROM user_performance WHERE user_id = $1::uuid",
                user_id,
            )
        return {row["topic"]: float(row["score"]) for row in rows}
    except Exception as e:
        logger.error(f"DB user performance fetch failed: {e}")
        return {}


async def _save_user_performance_to_db(
    user_id:       str,
    topic:         str,
    new_score:     float,
    answer_count:  int = 1,
) -> None:
    """
    Upsert a topic score into user_performance.
    Uses ON CONFLICT DO UPDATE to atomically increment answer_count.
    """
    pool = app_state.get("pool")
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_performance (user_id, topic, score, answer_count, updated_at)
                VALUES ($1::uuid, $2, $3, $4, NOW())
                ON CONFLICT (user_id, topic)
                DO UPDATE SET
                    score        = $3,
                    answer_count = user_performance.answer_count + $4,
                    updated_at   = NOW()
                """,
                user_id, topic, new_score, answer_count,
            )
    except Exception as e:
        logger.error(f"DB performance upsert failed: {e}")


async def _update_session_difficulty_in_db(
    session_id:         str,
    difficulty_level:   int,
    performance_score:  float,
) -> None:
    """Update the session's difficulty_level and performance_score in DB."""
    pool = app_state.get("pool")
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions
                SET difficulty_level  = $2,
                    performance_score = $3
                WHERE id = $1::uuid
                """,
                session_id, difficulty_level, performance_score,
            )
    except Exception as e:
        logger.error(f"DB session update failed: {e}")


# ── Question bank loader (with Redis cache) ───────────────────────────────────

async def _get_question_bank() -> list[QuestionRecord]:
    """
    Load active questions from Redis cache, falling back to DB on miss.
    Caches result for TTL_QUESTION_BANK seconds.
    """
    cache_key = "question_bank:active"
    cached = await _redis_get(cache_key)

    if cached:
        try:
            raw = json.loads(cached)
            return [QuestionRecord(**q) for q in raw]
        except Exception:
            pass  # corrupt cache → reload from DB

    questions = await _load_questions_from_db()

    if questions:
        await _redis_set(
            cache_key,
            json.dumps([vars(q) for q in questions]),
            TTL_QUESTION_BANK,
        )

    return questions


# ── Session state (Redis only) ────────────────────────────────────────────────

async def _load_session_context(session_id: str, user_id: str) -> SessionContext:
    """
    Load session context from Redis.
    Creates a fresh SessionContext if not found (first question of session).
    """
    key = f"session:{session_id}:state"
    cached = await _redis_get(key)

    if cached:
        try:
            data = json.loads(cached)
            return SessionContext(
                session_id          = data["session_id"],
                user_id             = data["user_id"],
                current_difficulty  = data.get("current_difficulty", 3.0),
                performance_score   = data.get("performance_score", 0.0),
                asked_question_ids  = data.get("asked_question_ids", []),
                category_counts     = data.get("category_counts", {}),
                total_questions     = data.get("total_questions", 0),
            )
        except Exception as e:
            logger.warning(f"Session state parse failed: {e}. Creating fresh context.")

    return SessionContext(session_id=session_id, user_id=user_id)


async def _save_session_context(context: SessionContext) -> None:
    """Persist session context back to Redis."""
    key = f"session:{context.session_id}:state"
    data = {
        "session_id":         context.session_id,
        "user_id":            context.user_id,
        "current_difficulty": context.current_difficulty,
        "performance_score":  context.performance_score,
        "asked_question_ids": context.asked_question_ids,
        "category_counts":    context.category_counts,
        "total_questions":    context.total_questions,
    }
    await _redis_set(key, json.dumps(data), TTL_SESSION)


# ── User performance loader (Redis + DB) ─────────────────────────────────────

async def _get_user_performance(user_id: str) -> dict[str, float]:
    """
    Load user's topic performance scores.
    Redis cache (1h TTL) → DB fallback.
    """
    key = f"user:{user_id}:performance"
    cached = await _redis_get(key)

    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    scores = await _load_user_performance_from_db(user_id)

    if scores:
        await _redis_set(key, json.dumps(scores), TTL_USER_PERF)

    return scores


async def _update_user_performance_cache(
    user_id:   str,
    topic:     str,
    new_score: float,
) -> None:
    """
    Update the Redis performance cache after a new answer.
    Write-through: update cache and DB.
    """
    key = f"user:{user_id}:performance"
    scores = await _get_user_performance(user_id)
    scores[topic] = new_score
    await _redis_set(key, json.dumps(scores), TTL_USER_PERF)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/next-question", response_model=NextQuestionResponse)
async def next_question(req: NextQuestionRequest) -> NextQuestionResponse:
    """
    Select the next question for a session based on current performance.

    Steps:
      1. Load question bank (cache or DB)
      2. Load session context (Redis)
      3. Merge in the session_question_ids from the request
         (client is authoritative for which questions were asked)
      4. Load user performance (cache or DB)
      5. Run adaptive selection
      6. Update session state + performance records
      7. Return selected question
    """
    if not app_state.get("ready"):
        raise HTTPException(status_code=503, detail="Service not ready")

    # ── Load question bank ────────────────────────────────────────────
    questions = await _get_question_bank()
    if not questions:
        raise HTTPException(
            status_code=503,
            detail={"error": "no_questions", "detail": "Question bank is empty"},
        )

    # ── Load session context ──────────────────────────────────────────
    context = await _load_session_context(req.session_id, req.user_id)

    # Merge client-supplied asked IDs (client is source of truth for session history)
    for qid in req.session_question_ids:
        if qid not in context.asked_question_ids:
            context.asked_question_ids.append(qid)

    # ── Load user performance ─────────────────────────────────────────
    topic_scores = await _get_user_performance(req.user_id)

    # ── Run adaptive selection ────────────────────────────────────────
    last_scores_dict = {
        "content":      req.last_scores.content,
        "relevance":    req.last_scores.relevance,
        "completeness": req.last_scores.completeness,
        "accuracy":     req.last_scores.accuracy,
    }

    result = select_next_question(
        questions    = questions,
        context      = context,
        topic_scores = topic_scores,
        last_scores  = last_scores_dict,
    )

    # ── Fallback if no unseen questions ──────────────────────────────
    if result is None:
        fallback_q = select_fallback_question(
            questions          = questions,
            asked_ids          = context.asked_question_ids,
            target_difficulty  = int(round(context.current_difficulty)),
        )
        if fallback_q is None:
            raise HTTPException(
                status_code=422,
                detail={"error": "session_complete", "detail": "No more questions available"},
            )
        result = SelectionResult(
            question_id      = fallback_q.id,
            question_text    = fallback_q.text,
            question_type    = fallback_q.type,
            difficulty       = fallback_q.difficulty,
            user_level       = round(context.current_difficulty, 2),
            selection_reason = "fallback_random",
        )

    # ── Update session state ──────────────────────────────────────────
    selected_q = next(
        (q for q in questions if q.id == result.question_id), None
    )
    if selected_q:
        context.category_counts[selected_q.type] = (
            context.category_counts.get(selected_q.type, 0) + 1
        )
        context.total_questions += 1

    await _save_session_context(context)

    # ── Update performance records (for the question just answered) ───
    # We update based on last_scores which reflect the PREVIOUS question's answer
    # The category to update is the last question's category (from session_question_ids[-1])
    if req.session_question_ids:
        last_q_id = req.session_question_ids[-1]
        last_q = next((q for q in questions if q.id == last_q_id), None)
        if last_q and last_q.category:
            overall_0_to_100 = (
                sum(last_scores_dict.values()) / (4 * 4)
            ) * 100  # 4 dimensions, max 4 each → 0-100

            current_topic_score = topic_scores.get(last_q.category, 50.0)
            new_topic_score = (
                EMA_ALPHA_PERF * overall_0_to_100
                + (1 - EMA_ALPHA_PERF) * current_topic_score
            )
            await _update_user_performance_cache(
                req.user_id, last_q.category, round(new_topic_score, 2)
            )
            await _save_user_performance_to_db(
                req.user_id, last_q.category, round(new_topic_score, 2)
            )

    # Also persist session difficulty to DB
    await _update_session_difficulty_in_db(
        req.session_id,
        int(round(context.current_difficulty)),
        context.performance_score,
    )

    return NextQuestionResponse(
        question_id      = result.question_id,
        question_text    = result.question_text,
        question_type    = result.question_type,
        difficulty       = result.difficulty,
        user_level       = result.user_level,
        selection_reason = result.selection_reason,
    )


@app.get("/performance/{user_id}", response_model=PerformanceResponse)
async def get_performance(user_id: str = Path(...)) -> PerformanceResponse:
    """
    Return a user's topic-by-topic performance summary.
    Reads from DB (authoritative), not Redis, for accuracy.
    """
    pool = app_state.get("pool")
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, score, answer_count
                FROM user_performance
                WHERE user_id = $1::uuid
                ORDER BY score ASC
                """,
                user_id,
            )
    except Exception as e:
        logger.error(f"Performance fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance data")

    if not rows:
        return PerformanceResponse(
            overall_score  = 0.0,
            topics         = {},
            priority_areas = [],
        )

    topics: dict[str, TopicStat] = {
        row["topic"]: TopicStat(
            score        = round(float(row["score"]), 1),
            answer_count = row["answer_count"],
        )
        for row in rows
    }

    scores = [t.score for t in topics.values()]
    overall = round(sum(scores) / len(scores), 1) if scores else 0.0

    # Priority areas = topics scored below average, sorted by score ascending
    avg = overall
    priority_areas = [
        topic for topic, stat in sorted(topics.items(), key=lambda x: x[1].score)
        if stat.score < avg
    ][:3]  # top 3 weakest

    return PerformanceResponse(
        overall_score  = overall,
        topics         = topics,
        priority_areas = priority_areas,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    pool   = app_state.get("pool")
    redis  = app_state.get("redis")

    db_ok = False
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    redis_ok = False
    if redis:
        try:
            await redis.ping()
            redis_ok = True
        except Exception:
            pass

    # Count questions in cache
    cached_count = 0
    raw = await _redis_get("question_bank:active")
    if raw:
        try:
            cached_count = len(json.loads(raw))
        except Exception:
            pass

    return HealthResponse(
        status           = "ok" if app_state.get("ready") else "starting",
        db               = db_ok,
        redis            = redis_ok,
        questions_cached = cached_count,
    )


# ── Module-level constant (needed by next_question endpoint) ─────────────────
EMA_ALPHA_PERF = 0.3  # same alpha as selector.py EMA_ALPHA
