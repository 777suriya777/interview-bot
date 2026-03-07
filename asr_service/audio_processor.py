"""
audio_processor.py — Audio pre-processing and paralinguistic feature extraction.

Implements the 9-stage ASR pipeline from the spec:
  1. Receive .webm from browser (handled in main.py UploadFile)
  2. Convert to .wav  — ffmpeg (subprocess) → 16kHz mono PCM
  3. Trim silence     — librosa.effects.trim(top_db=20)
  4. Noise reduction  — noisereduce.reduce_noise(stationary=True)
  5. Normalise        — librosa.util.normalize(norm=np.inf)
  6. (Whisper STT)    — handled by transcriber.py
  7. Extract MFCCs    — librosa.feature.mfcc(n_mfcc=40, hop_length=512)
  8. Compute paralinguistics — librosa.yin (pitch), custom WPM
  9. Generate delivery flags — rule engine (thresholds from spec)

Delivery flag thresholds (from spec Section 4.4):
  fast_speech:      WPM > 165
  nervous_pitch:    pitch_std > 35.0 Hz
  excessive_pauses: pause_count > 8 in < 3 min
  low_volume:       volume_variation < 0.05
  too_short:        word_count < 30
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import noisereduce as nr

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_SR          = 16_000    # Whisper expects 16kHz
SILENCE_TOP_DB     = 20        # librosa.effects.trim threshold
MFCC_N_COEF        = 40        # number of MFCC coefficients
MFCC_HOP_LENGTH    = 512       # ~32ms at 16kHz
PITCH_FMIN         = 75.0      # Hz — lowest expected human fundamental
PITCH_FMAX         = 400.0     # Hz — highest expected human fundamental
PAUSE_THRESHOLD_DB = -40.0     # dB below peak → silence
MIN_PAUSE_SECONDS  = 0.5       # gaps > 0.5s count as a pause
MIN_AUDIO_SECONDS  = 2.0       # reject audio shorter than this

# Delivery flag thresholds (exact values from spec)
WPM_FAST_THRESHOLD          = 165.0
PITCH_STD_NERVOUS_THRESHOLD = 35.0
PAUSE_COUNT_THRESHOLD       = 8      # more than this in < 3 min → excessive
VOLUME_VARIATION_LOW        = 0.05
WORD_COUNT_TOO_SHORT        = 30


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class AudioFeatures:
    """
    All extracted features from a single audio file.
    Matches the ASR Service response schema from the spec exactly.
    """
    # Duration
    duration_seconds: float = 0.0

    # Paralinguistic features
    speaking_rate_wpm:   float = 0.0
    pause_count:         int   = 0
    pitch_mean_hz:       float = 0.0
    pitch_std:           float = 0.0
    volume_variation:    float = 0.0

    # Delivery flags (bool — computed from thresholds)
    is_fast_speech:    bool = False
    is_nervous_pitch:  bool = False

    # Human-readable delivery warning strings
    delivery_flags: list[str] = field(default_factory=list)

    # MFCC array (not sent to client, used internally)
    mfcc_mean: Optional[np.ndarray] = field(default=None, repr=False)


# ── Stage 2: Convert to WAV ────────────────────────────────────────────────────

def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Convert any audio format (webm, mp4, ogg, …) to 16kHz mono PCM WAV.
    Uses ffmpeg subprocess — must be installed in the Docker image.

    Args:
        input_path:  path to the uploaded audio file
        output_path: path where the .wav will be written

    Raises:
        RuntimeError: if ffmpeg is not found or conversion fails
        FileNotFoundError: if input_path does not exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output without asking
        "-i", input_path,       # input file
        "-ar", str(TARGET_SR),  # resample to 16kHz
        "-ac", "1",             # mono channel
        "-f", "wav",            # output format
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,         # ffmpeg should never take >30s for short audio
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it: apt-get install -y ffmpeg"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out after 30 seconds")


# ── Stages 3–5: Pre-process audio ─────────────────────────────────────────────

def preprocess_audio(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Load a 16kHz mono WAV and apply:
      3. Silence trimming (top_db=20)
      4. Stationary noise reduction
      5. Amplitude normalisation (L∞ norm)

    Args:
        wav_path: path to the 16kHz mono .wav file

    Returns:
        (audio_array, sample_rate) — processed float32 numpy array

    Raises:
        ValueError: if audio is shorter than MIN_AUDIO_SECONDS after trimming
    """
    # Load as float32, force 16kHz mono
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # Stage 3: Trim leading/trailing silence
    # top_db=20 means samples < 20dB below the peak are treated as silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)

    duration = len(y_trimmed) / sr
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"Audio too short ({duration:.1f}s) after silence trimming. "
            f"Minimum is {MIN_AUDIO_SECONDS}s."
        )

    # Stage 4: Noise reduction
    # stationary=True uses the first ~1s of audio to estimate the noise profile
    # Works well for recording environments (fan, AC, room noise)
    try:
        y_denoised = nr.reduce_noise(
            y=y_trimmed,
            sr=sr,
            stationary=True,
            prop_decrease=0.75,  # reduce noise by 75%, not 100% to preserve naturalness
        )
    except Exception as e:
        # noisereduce can fail on very short or unusual audio — fall back gracefully
        logger.warning(f"Noise reduction failed ({e}), continuing with trimmed audio")
        y_denoised = y_trimmed

    # Stage 5: Amplitude normalisation (L∞ norm = peak normalisation)
    # Prevents librosa from having near-silent or clipping audio
    y_norm = librosa.util.normalize(y_denoised, norm=np.inf)

    logger.debug(f"Preprocessed audio: {duration:.2f}s → {len(y_norm)/sr:.2f}s after trim")
    return y_norm, sr


# ── Stage 7: MFCC extraction ───────────────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract MFCC features from preprocessed audio.
    Returns mean of each coefficient over time → shape (n_mfcc,).

    Args:
        y:  preprocessed float32 audio array
        sr: sample rate (should be TARGET_SR = 16000)

    Returns:
        numpy array of shape (MFCC_N_COEF,) = (40,)
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=MFCC_N_COEF,
        hop_length=MFCC_HOP_LENGTH,
    )
    # Mean over time — reduces (40, T) to (40,) for downstream use
    return mfcc.mean(axis=1)


# ── Stage 8: Paralinguistic feature extraction ────────────────────────────────

def extract_pitch_features(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimate fundamental frequency (F0) using YIN algorithm.
    Returns (pitch_mean_hz, pitch_std_hz) over voiced frames only.

    Voiced-frame filtering: frames with F0 < PITCH_FMIN or > PITCH_FMAX
    are treated as unvoiced and excluded from statistics. This prevents
    unvoiced frames (F0=0) from artificially deflating the mean and
    inflating the std.

    Args:
        y:  preprocessed float32 audio array
        sr: sample rate

    Returns:
        (pitch_mean_hz, pitch_std_hz) — 0.0 if no voiced frames found
    """
    try:
        f0 = librosa.yin(
            y,
            fmin=PITCH_FMIN,
            fmax=PITCH_FMAX,
            sr=sr,
            hop_length=MFCC_HOP_LENGTH,
        )
        # Filter to voiced frames (valid pitch estimates)
        voiced = f0[(f0 >= PITCH_FMIN) & (f0 <= PITCH_FMAX)]
        if len(voiced) == 0:
            return 0.0, 0.0
        return float(np.mean(voiced)), float(np.std(voiced))
    except Exception as e:
        logger.warning(f"Pitch extraction failed: {e}")
        return 0.0, 0.0


def extract_pause_count(y: np.ndarray, sr: int) -> int:
    """
    Count the number of pauses (silence gaps ≥ MIN_PAUSE_SECONDS).

    Method:
      1. Compute short-time RMS energy with a 512-sample frame
      2. Convert to dB, threshold at PAUSE_THRESHOLD_DB below peak
      3. Count contiguous below-threshold regions ≥ MIN_PAUSE_SECONDS

    Args:
        y:  preprocessed float32 audio array
        sr: sample rate

    Returns:
        Number of pause events detected
    """
    hop = MFCC_HOP_LENGTH

    # RMS energy per frame → dB
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Boolean mask: True = silence, False = speech
    silence_mask = rms_db < PAUSE_THRESHOLD_DB

    # Count transitions from speech→silence that last ≥ MIN_PAUSE_SECONDS
    min_frames = int(MIN_PAUSE_SECONDS * sr / hop)
    pause_count = 0
    consecutive_silence = 0
    in_pause = False

    for is_silent in silence_mask:
        if is_silent:
            consecutive_silence += 1
        else:
            if in_pause and consecutive_silence >= min_frames:
                pause_count += 1
            consecutive_silence = 0
            in_pause = False

        # Transition: enough silent frames to start a pause region
        if consecutive_silence >= min_frames and not in_pause:
            in_pause = True

    # Handle pause extending to end of audio
    if in_pause and consecutive_silence >= min_frames:
        pause_count += 1

    return pause_count


def extract_volume_variation(y: np.ndarray, sr: int) -> float:
    """
    Compute volume variation = std(RMS) / (mean(RMS) + ε).

    A flat value < 0.05 suggests monotone delivery (low_volume flag).
    A higher value indicates natural prosodic variation.

    Returns:
        float between 0.0 and ~1.0
    """
    rms = librosa.feature.rms(y=y, hop_length=MFCC_HOP_LENGTH)[0]
    mean_rms = float(np.mean(rms))
    std_rms  = float(np.std(rms))
    return round(std_rms / (mean_rms + 1e-9), 4)


def compute_wpm(transcript: str, duration_seconds: float) -> float:
    """
    Compute speaking rate in words per minute.

    Args:
        transcript:       the Whisper transcript string
        duration_seconds: audio duration after silence trimming

    Returns:
        WPM as float, 0.0 if duration is zero or transcript is empty
    """
    if not transcript or duration_seconds <= 0:
        return 0.0
    word_count = len(transcript.split())
    minutes = duration_seconds / 60.0
    return round(word_count / minutes, 2)


# ── Stage 9: Delivery flag generation ─────────────────────────────────────────

def generate_delivery_flags(
    wpm:              float,
    pitch_std:        float,
    pause_count:      int,
    volume_variation: float,
    word_count:       int,
    duration_seconds: float,
) -> list[str]:
    """
    Apply threshold rules from spec Table (Section 4.4) to produce
    human-readable delivery warning strings.

    Returns:
        List of flag strings (empty list = no issues detected)
    """
    flags: list[str] = []

    # fast_speech: WPM > 165
    if wpm > WPM_FAST_THRESHOLD:
        flags.append(
            f"You spoke at {wpm:.0f} WPM — try slowing to 130–150 WPM."
        )

    # nervous_pitch: pitch_std > 35.0 Hz
    if pitch_std > PITCH_STD_NERVOUS_THRESHOLD:
        flags.append(
            "Significant pitch variation detected; try speaking more evenly."
        )

    # excessive_pauses: pause_count > 8 in < 3 min
    if pause_count > PAUSE_COUNT_THRESHOLD and duration_seconds < 180:
        flags.append(
            "Frequent pausing — use filler phrases less and practise transitions."
        )

    # low_volume: volume_variation < 0.05
    if volume_variation < VOLUME_VARIATION_LOW:
        flags.append(
            "Your voice level was very flat — vary your emphasis to sound engaging."
        )

    # too_short: word_count < 30
    if word_count < WORD_COUNT_TOO_SHORT:
        flags.append(
            "Your answer was very brief — aim for at least 60–90 words."
        )

    return flags


# ── Full pipeline (orchestrator) ───────────────────────────────────────────────

def process_audio_file(
    input_path: str,
    transcript: str,
    wav_output_path: Optional[str] = None,
) -> AudioFeatures:
    """
    Run the full audio feature extraction pipeline on an uploaded audio file.

    Stages covered: 2 (ffmpeg), 3–5 (preprocess), 7 (MFCCs), 8 (paralinguistics), 9 (flags).
    Stage 6 (Whisper STT) is handled separately by transcriber.py.

    Args:
        input_path:      path to the uploaded audio file (.webm, .wav, …)
        transcript:      the Whisper transcript (needed for WPM calculation)
        wav_output_path: optional path to save the processed .wav; uses a
                         temp file if not provided

    Returns:
        AudioFeatures dataclass with all computed values
    """
    features = AudioFeatures()

    # ── Stage 2: Convert to WAV ──────────────────────────────────────
    use_temp = wav_output_path is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
    else:
        wav_path = wav_output_path

    try:
        convert_to_wav(input_path, wav_path)

        # ── Stages 3–5: Pre-process ─────────────────────────────────
        y, sr = preprocess_audio(wav_path)
        features.duration_seconds = round(len(y) / sr, 2)

        # ── Stage 7: MFCCs ──────────────────────────────────────────
        try:
            features.mfcc_mean = extract_mfcc(y, sr)
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")

        # ── Stage 8: Paralinguistics ─────────────────────────────────
        pitch_mean, pitch_std = extract_pitch_features(y, sr)
        features.pitch_mean_hz = round(pitch_mean, 2)
        features.pitch_std     = round(pitch_std, 4)

        features.pause_count      = extract_pause_count(y, sr)
        features.volume_variation = extract_volume_variation(y, sr)

        word_count = len(transcript.split()) if transcript else 0
        features.speaking_rate_wpm = compute_wpm(transcript, features.duration_seconds)

        # Convenience booleans for the response schema
        features.is_fast_speech   = features.speaking_rate_wpm > WPM_FAST_THRESHOLD
        features.is_nervous_pitch = features.pitch_std > PITCH_STD_NERVOUS_THRESHOLD

        # ── Stage 9: Delivery flags ───────────────────────────────────
        features.delivery_flags = generate_delivery_flags(
            wpm              = features.speaking_rate_wpm,
            pitch_std        = features.pitch_std,
            pause_count      = features.pause_count,
            volume_variation = features.volume_variation,
            word_count       = word_count,
            duration_seconds = features.duration_seconds,
        )

    finally:
        # Clean up temp WAV file
        if use_temp and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    return features
