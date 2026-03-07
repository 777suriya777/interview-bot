"""
transcriber.py — Whisper ASR wrapper.

Wraps openai-whisper with:
  - Model loaded once at startup (singleton pattern)
  - Configurable model size via WHISPER_MODEL_SIZE env var (base|small|medium)
  - Language forced to English for consistent WER
  - fp16=False on CPU (as spec notes) — float16 requires CUDA

Model sizes and approximate WER from spec:
  base:   fastest, higher WER
  small:  spec default — 5.3% overall WER, 3.2% clear speech (validated result)
  medium: better accuracy but slower inference

The transcriber does NOT handle audio pre-processing.
It expects a clean, pre-processed .wav file from audio_processor.py.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level model storage — loaded once in load_model(), reused per request
_whisper_model = None
_model_size: str = ""


def load_model(model_size: Optional[str] = None) -> None:
    """
    Load the Whisper model into module-level cache.
    Call this once at service startup via FastAPI lifespan.

    Args:
        model_size: 'base' | 'small' | 'medium' — defaults to WHISPER_MODEL_SIZE env var,
                    falls back to 'small' (spec default).
    """
    global _whisper_model, _model_size

    import whisper  # import here to avoid top-level import when testing without whisper

    size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "small")
    _model_size = size

    logger.info(f"Loading Whisper model: {size}…")
    try:
        _whisper_model = whisper.load_model(size)
        logger.info(f"✓ Whisper '{size}' model loaded")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise


def transcribe(wav_path: str) -> dict:
    """
    Transcribe a 16kHz mono .wav file using the cached Whisper model.

    Args:
        wav_path: path to the pre-processed .wav file

    Returns:
        dict with keys:
          - transcript (str): the full transcription text
          - language (str):   detected language code (usually 'en')
          - segments (list):  list of timed segment dicts (not sent to client)

    Raises:
        RuntimeError: if model not loaded (load_model() not called)
        FileNotFoundError: if wav_path does not exist
    """
    if _whisper_model is None:
        raise RuntimeError(
            "Whisper model not loaded. Call transcriber.load_model() at startup."
        )

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    logger.debug(f"Transcribing: {wav_path}")
    try:
        result = _whisper_model.transcribe(
            wav_path,
            language="en",      # force English for consistent WER
            fp16=False,         # must be False on CPU — fp16 requires CUDA
            verbose=False,      # suppress per-segment stdout
        )
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        raise RuntimeError(f"Transcription failed: {e}") from e

    transcript = result.get("text", "").strip()
    logger.debug(f"Transcript ({len(transcript.split())} words): {transcript[:80]}…")

    return {
        "transcript": transcript,
        "language":   result.get("language", "en"),
        "segments":   result.get("segments", []),
    }


def get_model_info() -> dict:
    """Return info about the currently loaded model for the /health endpoint."""
    return {
        "loaded":     _whisper_model is not None,
        "model_size": _model_size or "not_loaded",
    }
