"""
ASR Service — FastAPI endpoint for speech-to-text and paralinguistic analysis.

Endpoints:
  POST /transcribe  (multipart/form-data, field: audio_file)
  GET  /health

Processing order (matches spec Section 5.1):
  1. Receive .webm or .wav upload (max MAX_AUDIO_MB)
  2. Save to temp file
  3. Convert to 16kHz mono WAV via ffmpeg
  4–5. Pre-process (silence trim, noise reduction, normalisation)
  6. Whisper STT → transcript
  7–8. Extract MFCC + paralinguistic features (WPM, pitch, pauses)
  9. Generate delivery flags

Error codes:
  too_short        → audio < 2s after trimming (HTTP 422)
  file_too_large   → exceeds MAX_AUDIO_MB (HTTP 413)
  transcribe_error → Whisper or ffmpeg failure (HTTP 500)
"""
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import audio_processor as ap
import transcriber

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)

MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_MB", "10")) * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/x-wav",
    "audio/mp4", "audio/mpeg", "video/webm",  # browsers sometimes send video/webm
}

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Whisper model once at startup."""
    logger.info("ASR service starting up…")
    try:
        transcriber.load_model()
        logger.info("✓ Whisper model ready")
    except Exception as e:
        # Service still starts — health endpoint will report not loaded.
        # This allows the container to start even if the model download fails,
        # which is preferable to a crash loop in dev.
        logger.error(f"Whisper model load failed: {e}. Service will return 503 until fixed.")
    yield
    logger.info("ASR service shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Bot — ASR Service",
    description="Whisper speech-to-text + paralinguistic feature extraction",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript:        str
    speaking_rate_wpm: float
    pause_count:       int
    pitch_mean_hz:     float
    pitch_std:         float
    volume_variation:  float
    is_fast_speech:    bool
    is_nervous_pitch:  bool
    delivery_flags:    list[str]


class HealthResponse(BaseModel):
    status:      str
    model:       str
    loaded:      bool
    model_size:  str


# ── Helper ────────────────────────────────────────────────────────────────────

async def _save_upload_to_temp(file: UploadFile) -> tuple[str, str]:
    """
    Read the uploaded file, validate size, save to a temp file.

    Returns:
        (temp_path, suffix) where suffix is the file extension (.webm, .wav, etc.)

    Raises:
        HTTPException 413 if file exceeds MAX_AUDIO_MB
        HTTPException 415 if content type is not an audio format
    """
    # Determine extension from filename or content_type
    filename = file.filename or "audio.webm"
    suffix = os.path.splitext(filename)[-1].lower()
    if not suffix:
        suffix = ".webm"  # default for browser recordings

    # Read entire file, then enforce size limit.
    # Starlette UploadFile uses .read(), not async iteration.
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": f"Audio file exceeds {MAX_AUDIO_BYTES // (1024*1024)}MB limit",
            },
        )

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()

    return tmp.name, suffix


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(
    audio_file: UploadFile = File(..., description="Audio file (.webm or .wav, max 10MB)"),
) -> TranscribeResponse:
    """
    Transcribe a voice answer and extract paralinguistic features.

    Steps:
      1. Save upload → temp file
      2. Convert to 16kHz WAV (ffmpeg)
      3. Pre-process (trim, denoise, normalise)
      4. Whisper STT
      5. Extract features (WPM, pitch, pauses, MFCC)
      6. Generate delivery flags
      7. Return structured response
    """
    model_info = transcriber.get_model_info()
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "detail": "Whisper model not loaded"},
        )

    # Save upload to disk
    input_path: Optional[str] = None
    try:
        input_path, _ = await _save_upload_to_temp(audio_file)

        # Run Whisper transcription on the raw upload first
        # (Whisper can handle webm directly; we also run ffmpeg for feature extraction)
        try:
            result = transcriber.transcribe(input_path)
            transcript = result["transcript"]
        except RuntimeError as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "transcribe_error", "detail": str(e)},
            )

        # Extract audio features from the same file
        try:
            features = ap.process_audio_file(
                input_path  = input_path,
                transcript  = transcript,
            )
        except ValueError as e:
            # Raised by preprocess_audio when audio is too short
            raise HTTPException(
                status_code=422,
                detail={"error": "too_short", "detail": str(e)},
            )
        except RuntimeError as e:
            # ffmpeg error or other processing failure
            logger.error(f"Audio processing error: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": "processing_error", "detail": str(e)},
            )

        return TranscribeResponse(
            transcript        = transcript,
            speaking_rate_wpm = features.speaking_rate_wpm,
            pause_count       = features.pause_count,
            pitch_mean_hz     = features.pitch_mean_hz,
            pitch_std         = features.pitch_std,
            volume_variation  = features.volume_variation,
            is_fast_speech    = features.is_fast_speech,
            is_nervous_pitch  = features.is_nervous_pitch,
            delivery_flags    = features.delivery_flags,
        )

    finally:
        # Always clean up the temp upload file
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe for Docker healthcheck."""
    info = transcriber.get_model_info()
    return HealthResponse(
        status     = "ok" if info["loaded"] else "degraded",
        model      = "openai-whisper",
        loaded     = info["loaded"],
        model_size = info["model_size"],
    )
