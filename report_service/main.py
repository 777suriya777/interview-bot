"""
Report Service — FastAPI service for session report generation.

Endpoints:
  POST /generate              → aggregate session data, render PDF, return URL
  GET  /report/{session_id}   → return pre-built report data (JSON, no PDF)
  GET  /health                → liveness probe

Report pipeline:
  1. Receive session_id via POST /generate
  2. Query PostgreSQL: session + answers + questions (single join query)
  3. Aggregate metrics: per-topic scores, confidence distribution, improvement areas
  4. Render Jinja2 HTML template with aggregated data
  5. Convert HTML → PDF with WeasyPrint
  6. Store PDF:
     - If AWS_ACCESS_KEY_ID set: upload to S3, return public URL
     - Otherwise: save to REPORT_LOCAL_DIR, return local path URL
  7. Update sessions.report_url in DB
  8. Return report JSON + PDF URL

All DB access is via asyncpg (not SQLAlchemy ORM — keeps service lightweight).
"""
from __future__ import annotations

import logging
import os
import tempfile
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from weasyprint import HTML as WeasyHTML

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)

# ── Config ────────────────────────────────────────────────────────────────────
TEMPLATE_DIR     = Path(__file__).parent / "templates"
REPORT_LOCAL_DIR = Path(os.environ.get("REPORT_LOCAL_DIR", "/tmp/reports"))
REPORT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
app_state: dict = {}

# ── Jinja2 environment ────────────────────────────────────────────────────────
jinja_env = Environment(
    loader        = FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape    = select_autoescape(["html"]),
    trim_blocks   = True,
    lstrip_blocks = True,
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Report service starting up…")

    db_url = os.environ.get("DATABASE_URL", "")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    pool = None
    try:
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        logger.info("✓ PostgreSQL pool created")
    except Exception as e:
        logger.error(f"DB pool creation failed: {e}")

    app_state["pool"]  = pool
    app_state["ready"] = True
    yield

    logger.info("Report service shutting down.")
    if pool:
        await pool.close()
    app_state.clear()


app = FastAPI(
    title       = "Interview Bot — Report Service",
    description = "Session report aggregation and PDF generation",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    session_id: str


class AnswerDetail(BaseModel):
    question_text:      str
    difficulty:         int
    transcript:         Optional[str]
    content_score:      int
    relevance_score:    int
    completeness_score: int
    accuracy_score:     int
    overall_score:      float
    confidence_label:   str
    delivery_flags:     list[str]


class ReportResponse(BaseModel):
    session_id:              str
    duration_seconds:        int
    total_questions:         int
    average_overall_score:   float
    per_topic_scores:        dict[str, float]
    confidence_distribution: dict[str, int]
    improvement_areas:       list[str]
    report_pdf_url:          str


class HealthResponse(BaseModel):
    status: str
    db:     bool


# ── Data aggregation ───────────────────────────────────────────────────────────

async def _fetch_session_data(session_id: str) -> dict[str, Any]:
    """
    Fetch all session data in one query join.
    Returns a dict with session metadata + list of answer rows.

    Raises HTTPException 404 if session not found, 500 on DB error.
    """
    pool = app_state.get("pool")
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            # Fetch session metadata
            session_row = await conn.fetchrow(
                """
                SELECT
                    s.id::text           AS session_id,
                    s.interview_type,
                    s.target_role,
                    s.started_at,
                    s.ended_at,
                    s.difficulty_level,
                    u.name               AS candidate_name
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.id = $1::uuid
                """,
                session_id,
            )

            if not session_row:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

            # Fetch all answers with their question text and category
            answer_rows = await conn.fetch(
                """
                SELECT
                    a.id::text                   AS answer_id,
                    q.text                       AS question_text,
                    COALESCE(q.difficulty, 3)    AS difficulty,
                    COALESCE(q.category, '')     AS category,
                    a.transcript,
                    COALESCE(a.content_score,      0) AS content_score,
                    COALESCE(a.relevance_score,    0) AS relevance_score,
                    COALESCE(a.completeness_score, 0) AS completeness_score,
                    COALESCE(a.accuracy_score,     0) AS accuracy_score,
                    COALESCE(a.overall_score,      0) AS overall_score,
                    COALESCE(a.confidence_label, 'moderate') AS confidence_label,
                    a.delivery_flags,
                    a.submitted_at
                FROM answers a
                JOIN questions q ON q.id = a.question_id
                WHERE a.session_id = $1::uuid
                ORDER BY a.submitted_at ASC
                """,
                session_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB fetch failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"session": dict(session_row), "answers": [dict(r) for r in answer_rows]}


def _aggregate_report(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate raw DB rows into the report data structure.

    Computes:
      - duration_seconds
      - average_overall_score
      - per_topic_scores: mean overall_score per category
      - confidence_distribution: count per label
      - improvement_areas: topics with below-average scores (max 3)
    """
    session = raw["session"]
    answers = raw["answers"]

    # ── Duration ────────────────────────────────────────────────
    started_at = session.get("started_at")
    ended_at   = session.get("ended_at") or datetime.now(tz=timezone.utc)
    if started_at and ended_at:
        duration_seconds = int((ended_at - started_at).total_seconds())
    else:
        duration_seconds = 0

    # ── Score aggregation ────────────────────────────────────────
    overall_scores = [float(a["overall_score"]) for a in answers if a["overall_score"] is not None]
    avg_overall    = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0.0

    # Per-dimension averages
    dims = ["content_score", "relevance_score", "completeness_score", "accuracy_score"]
    avg_scores = {}
    for dim in dims:
        vals = [float(a[dim]) for a in answers if a[dim] is not None]
        avg_scores[dim.replace("_score", "")] = round(sum(vals) / len(vals), 2) if vals else 0.0

    # ── Per-topic scores ─────────────────────────────────────────
    topic_buckets: dict[str, list[float]] = {}
    for a in answers:
        cat = a.get("category") or "general"
        if cat not in topic_buckets:
            topic_buckets[cat] = []
        if a["overall_score"] is not None:
            topic_buckets[cat].append(float(a["overall_score"]))

    per_topic_scores: dict[str, float] = {
        topic: round(sum(scores) / len(scores), 2)
        for topic, scores in topic_buckets.items()
        if scores
    }

    # ── Confidence distribution ──────────────────────────────────
    conf_dist = {"high": 0, "moderate": 0, "low": 0, "anxious": 0}
    for a in answers:
        label = a.get("confidence_label") or "moderate"
        if label in conf_dist:
            conf_dist[label] += 1

    # ── Improvement areas ────────────────────────────────────────
    if per_topic_scores:
        avg_topic = sum(per_topic_scores.values()) / len(per_topic_scores)
        improvement_areas = sorted(
            [t for t, s in per_topic_scores.items() if s < avg_topic],
            key=lambda t: per_topic_scores[t],
        )[:3]
    else:
        improvement_areas = []

    # ── Duration string for template ─────────────────────────────
    mins = duration_seconds // 60
    secs = duration_seconds % 60
    duration_str = f"{mins}m {secs:02d}s"

    # ── Answer details for template ──────────────────────────────
    answer_details = []
    for a in answers:
        flags_raw = a.get("delivery_flags") or []
        # delivery_flags may be stored as JSON string or already a list
        if isinstance(flags_raw, str):
            try:
                flags_raw = json.loads(flags_raw)
            except Exception:
                flags_raw = []
        answer_details.append({
            "question_text":      a["question_text"],
            "difficulty":         a["difficulty"],
            "transcript":         a.get("transcript") or "",
            "content_score":      int(a["content_score"]),
            "relevance_score":    int(a["relevance_score"]),
            "completeness_score": int(a["completeness_score"]),
            "accuracy_score":     int(a["accuracy_score"]),
            "overall_score":      float(a["overall_score"]),
            "confidence_label":   a["confidence_label"],
            "delivery_flags":     flags_raw if isinstance(flags_raw, list) else [],
        })

    return {
        # Report data (returned as JSON to client)
        "session_id":              session["session_id"],
        "interview_type":          session["interview_type"],
        "target_role":             session.get("target_role") or "",
        "duration_seconds":        duration_seconds,
        "total_questions":         len(answers),
        "average_overall_score":   avg_overall,
        "avg_scores":              avg_scores,
        "per_topic_scores":        per_topic_scores,
        "confidence_distribution": conf_dist,
        "improvement_areas":       improvement_areas,
        # Template-only fields
        "candidate_name":  session.get("candidate_name") or "Candidate",
        "duration_str":    duration_str,
        "session_date":    (started_at.strftime("%d %b %Y") if started_at else "—"),
        "answers":         answer_details,
    }


# ── PDF generation ────────────────────────────────────────────────────────────

def _render_pdf(report_data: dict[str, Any]) -> bytes:
    """
    Render the Jinja2 HTML template and convert to PDF with WeasyPrint.

    Args:
        report_data: aggregated report dict (used as template context)

    Returns:
        PDF bytes

    Raises:
        RuntimeError on template or WeasyPrint error
    """
    try:
        template = jinja_env.get_template("report.html")
        html_str = template.render(**report_data)
    except Exception as e:
        raise RuntimeError(f"Template rendering failed: {e}") from e

    try:
        pdf_bytes = WeasyHTML(string=html_str).write_pdf()
    except Exception as e:
        raise RuntimeError(f"WeasyPrint PDF generation failed: {e}") from e

    return pdf_bytes


def _store_pdf_local(session_id: str, pdf_bytes: bytes) -> str:
    """
    Save PDF to REPORT_LOCAL_DIR and return a local file URL.

    Returns:
        URL string like "file:///tmp/reports/<session_id>.pdf"
    """
    filename = f"{session_id}.pdf"
    pdf_path = REPORT_LOCAL_DIR / filename
    pdf_path.write_bytes(pdf_bytes)
    logger.info(f"Report saved locally: {pdf_path}")
    return f"file://{pdf_path}"


async def _store_pdf_s3(session_id: str, pdf_bytes: bytes) -> str:
    """
    Upload PDF to S3 and return the public URL.

    Falls back to local storage if boto3 is not installed or credentials are missing.

    Returns:
        S3 public URL or local file URL fallback
    """
    bucket = os.environ.get("S3_BUCKET", "interview-bot-reports")
    region = os.environ.get("S3_REGION", "ap-south-1")
    key    = f"reports/{session_id}.pdf"

    try:
        import boto3  # optional dependency — only needed in prod
        from botocore.exceptions import BotoCoreError, ClientError

        s3 = boto3.client("s3", region_name=region)
        s3.put_object(
            Bucket      = bucket,
            Key         = key,
            Body        = pdf_bytes,
            ContentType = "application/pdf",
        )
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        logger.info(f"Report uploaded to S3: {url}")
        return url

    except ImportError:
        logger.warning("boto3 not installed — storing report locally")
        return _store_pdf_local(session_id, pdf_bytes)
    except Exception as e:
        logger.error(f"S3 upload failed: {e}. Falling back to local storage.")
        return _store_pdf_local(session_id, pdf_bytes)


async def _save_report_url_to_db(session_id: str, report_url: str) -> None:
    """Persist the report URL back to sessions.report_url."""
    pool = app_state.get("pool")
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions
                SET report_url = $2,
                    status     = 'completed',
                    ended_at   = COALESCE(ended_at, NOW())
                WHERE id = $1::uuid
                """,
                session_id, report_url,
            )
    except Exception as e:
        logger.error(f"Failed to save report URL for session {session_id}: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=ReportResponse)
async def generate_report(req: GenerateRequest) -> ReportResponse:
    """
    Generate a PDF performance report for a completed session.

    Steps:
      1. Fetch session + answers from DB
      2. Aggregate metrics
      3. Render Jinja2 template
      4. Convert to PDF (WeasyPrint)
      5. Store PDF (S3 if configured, else local)
      6. Update sessions.report_url
      7. Return report JSON

    Idempotent: calling twice for the same session overwrites the previous PDF.
    """
    if not app_state.get("ready"):
        raise HTTPException(status_code=503, detail="Service not ready")

    # ── Fetch & aggregate ────────────────────────────────────────
    raw         = await _fetch_session_data(req.session_id)
    report_data = _aggregate_report(raw)

    # ── Generate PDF ─────────────────────────────────────────────
    try:
        pdf_bytes = _render_pdf(report_data)
    except RuntimeError as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # ── Store PDF ─────────────────────────────────────────────────
    use_s3 = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
    if use_s3:
        report_url = await _store_pdf_s3(req.session_id, pdf_bytes)
    else:
        report_url = _store_pdf_local(req.session_id, pdf_bytes)

    # ── Persist URL ───────────────────────────────────────────────
    await _save_report_url_to_db(req.session_id, report_url)

    return ReportResponse(
        session_id              = report_data["session_id"],
        duration_seconds        = report_data["duration_seconds"],
        total_questions         = report_data["total_questions"],
        average_overall_score   = report_data["average_overall_score"],
        per_topic_scores        = report_data["per_topic_scores"],
        confidence_distribution = report_data["confidence_distribution"],
        improvement_areas       = report_data["improvement_areas"],
        report_pdf_url          = report_url,
    )


@app.get("/report/{session_id}", response_model=ReportResponse)
async def get_report(session_id: str = FPath(...)) -> ReportResponse:
    """
    Return report data (JSON) for a session.
    Does NOT regenerate the PDF — reads existing report_url from DB.

    Used by GET /api/sessions/:id/report on the chatbot engine.
    """
    if not app_state.get("ready"):
        raise HTTPException(status_code=503, detail="Service not ready")

    raw         = await _fetch_session_data(session_id)
    report_data = _aggregate_report(raw)

    # Fetch the stored report URL from the sessions table
    pool = app_state.get("pool")
    report_url = ""
    if pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT report_url FROM sessions WHERE id = $1::uuid",
                    session_id,
                )
                if row:
                    report_url = row["report_url"] or ""
        except Exception as e:
            logger.warning(f"Could not fetch report_url: {e}")

    return ReportResponse(
        session_id              = report_data["session_id"],
        duration_seconds        = report_data["duration_seconds"],
        total_questions         = report_data["total_questions"],
        average_overall_score   = report_data["average_overall_score"],
        per_topic_scores        = report_data["per_topic_scores"],
        confidence_distribution = report_data["confidence_distribution"],
        improvement_areas       = report_data["improvement_areas"],
        report_pdf_url          = report_url,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    pool  = app_state.get("pool")
    db_ok = False
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    return HealthResponse(
        status = "ok" if app_state.get("ready") else "starting",
        db     = db_ok,
    )
