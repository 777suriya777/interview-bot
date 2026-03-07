"""
Tests for the ASR Service.

Groups:
  TestComputeWPM             → audio_processor.compute_wpm()
  TestGenerateDeliveryFlags  → audio_processor.generate_delivery_flags()
  TestExtractVolumeVariation → audio_processor.extract_volume_variation()
  TestPauseCount             → audio_processor.extract_pause_count()
  TestConvertToWav           → audio_processor.convert_to_wav() (mocked ffmpeg)
  TestTranscribeEndpoint     → POST /transcribe (mocked Whisper + librosa)
  TestHealthEndpoint         → GET /health

No real audio files, ffmpeg, or Whisper are required.
"""
import os
import sys
import types
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import numpy as np

# Make asr_service importable
ASR_DIR = str(Path(__file__).parent.parent / "asr_service")
if ASR_DIR not in sys.path:
    sys.path.insert(0, ASR_DIR)


# ── Stub heavy dependencies before any import ─────────────────────────────────
# librosa and noisereduce are installed in Docker but may not be in the test env.

def _ensure_librosa_stub():
    if "librosa" not in sys.modules:
        lib = types.ModuleType("librosa")
        lib.load   = MagicMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))
        lib.effects = types.SimpleNamespace(
            trim=lambda y, top_db=20: (y, (0, len(y)))
        )
        lib.util    = types.SimpleNamespace(normalize=lambda y, norm=None: y)
        lib.feature = types.SimpleNamespace(
            mfcc=lambda y, sr, n_mfcc, hop_length: np.zeros((n_mfcc, 10)),
            rms =lambda y, hop_length: np.array([[0.1] * 20]),
        )
        lib.yin     = MagicMock(return_value=np.array([120.0, 130.0, 125.0, 0.0]))
        lib.amplitude_to_db = lambda x, ref=None: x * 10
        sys.modules["librosa"] = lib

    if "noisereduce" not in sys.modules:
        nr = types.ModuleType("noisereduce")
        nr.reduce_noise = lambda y, sr, **kw: y
        sys.modules["noisereduce"] = nr

_ensure_librosa_stub()


# ── TestComputeWPM ─────────────────────────────────────────────────────────────

class TestComputeWPM:

    def test_basic_wpm(self):
        from audio_processor import compute_wpm
        # 150 words in 60 seconds → 150 WPM
        transcript = " ".join(["word"] * 150)
        assert compute_wpm(transcript, 60.0) == pytest.approx(150.0, rel=0.01)

    def test_fast_speech(self):
        from audio_processor import compute_wpm
        # 200 words in 60 seconds → 200 WPM
        transcript = " ".join(["word"] * 200)
        result = compute_wpm(transcript, 60.0)
        assert result > 165.0

    def test_empty_transcript_returns_zero(self):
        from audio_processor import compute_wpm
        assert compute_wpm("", 60.0) == 0.0

    def test_zero_duration_returns_zero(self):
        from audio_processor import compute_wpm
        assert compute_wpm("hello world", 0.0) == 0.0

    def test_returns_float(self):
        from audio_processor import compute_wpm
        result = compute_wpm("hello world this is a test", 10.0)
        assert isinstance(result, float)

    def test_short_answer_wpm(self):
        from audio_processor import compute_wpm
        # 30 words in 30 seconds = 60 WPM
        transcript = " ".join(["word"] * 30)
        result = compute_wpm(transcript, 30.0)
        assert result == pytest.approx(60.0, rel=0.01)


# ── TestGenerateDeliveryFlags ─────────────────────────────────────────────────

class TestGenerateDeliveryFlags:

    def _flags(self, **kwargs):
        from audio_processor import generate_delivery_flags
        defaults = {
            "wpm": 140.0,
            "pitch_std": 20.0,
            "pause_count": 3,
            "volume_variation": 0.15,
            "word_count": 80,
            "duration_seconds": 60.0,
        }
        defaults.update(kwargs)
        return generate_delivery_flags(**defaults)

    def test_no_flags_normal_answer(self):
        assert self._flags() == []

    def test_fast_speech_flag(self):
        flags = self._flags(wpm=180.0)
        assert any("WPM" in f or "slowing" in f.lower() for f in flags)

    def test_nervous_pitch_flag(self):
        flags = self._flags(pitch_std=40.0)
        assert any("pitch" in f.lower() for f in flags)

    def test_excessive_pauses_flag(self):
        flags = self._flags(pause_count=9, duration_seconds=120.0)
        assert any("paus" in f.lower() for f in flags)

    def test_excessive_pauses_not_triggered_over_3min(self):
        # >8 pauses but duration > 180s → should NOT trigger
        flags = self._flags(pause_count=10, duration_seconds=200.0)
        assert not any("paus" in f.lower() for f in flags)

    def test_low_volume_flag(self):
        flags = self._flags(volume_variation=0.03)
        assert any("flat" in f.lower() or "volume" in f.lower() for f in flags)

    def test_too_short_flag(self):
        flags = self._flags(word_count=15)
        assert any("brief" in f.lower() or "word" in f.lower() for f in flags)

    def test_multiple_flags(self):
        flags = self._flags(wpm=200.0, pitch_std=50.0, word_count=10)
        assert len(flags) >= 2

    def test_returns_list_of_strings(self):
        flags = self._flags(wpm=180.0)
        assert isinstance(flags, list)
        assert all(isinstance(f, str) for f in flags)

    def test_exactly_at_wpm_threshold_no_flag(self):
        # WPM == 165 exactly → should NOT trigger (threshold is strictly >165)
        flags = self._flags(wpm=165.0)
        assert not any("WPM" in f for f in flags)

    def test_just_above_wpm_threshold_triggers(self):
        flags = self._flags(wpm=165.1)
        assert any("WPM" in f for f in flags)


# ── TestExtractVolumeVariation ────────────────────────────────────────────────

class TestExtractVolumeVariation:

    def test_returns_float(self):
        from audio_processor import extract_volume_variation
        y = np.random.rand(16000).astype(np.float32)
        with patch("audio_processor.librosa") as mock_lib:
            mock_lib.feature.rms.return_value = np.array([[0.1, 0.2, 0.1, 0.3, 0.2]])
            mock_lib.feature.rms.__call__ = mock_lib.feature.rms
            result = extract_volume_variation(y, 16000)
        assert isinstance(result, float)

    def test_monotone_returns_low_value(self):
        from audio_processor import extract_volume_variation
        # Uniform RMS → std=0 → variation=0
        y = np.ones(16000, dtype=np.float32) * 0.1
        with patch("audio_processor.librosa") as mock_lib:
            mock_lib.feature.rms.return_value = np.array([[0.1] * 20])
            result = extract_volume_variation(y, 16000)
        assert result < 0.05  # should be at or near 0

    def test_varied_volume_returns_higher_value(self):
        from audio_processor import extract_volume_variation
        y = np.random.rand(16000).astype(np.float32)
        with patch("audio_processor.librosa") as mock_lib:
            mock_lib.feature.rms.return_value = np.array([[0.01, 0.5, 0.01, 0.5, 0.3, 0.01]])
            result = extract_volume_variation(y, 16000)
        assert result > 0.05


# ── TestPauseCount ────────────────────────────────────────────────────────────

class TestPauseCount:

    def test_no_pauses(self):
        from audio_processor import extract_pause_count
        y = np.ones(32000, dtype=np.float32) * 0.5  # continuous audio
        with patch("audio_processor.librosa") as mock_lib:
            # RMS is consistently high → all frames are speech
            mock_lib.feature.rms.return_value = np.array([[0.5] * 100])
            mock_lib.amplitude_to_db.return_value = np.array([-5.0] * 100)
            result = extract_pause_count(y, 16000)
        assert result == 0

    def test_pauses_detected(self):
        from audio_processor import extract_pause_count
        y = np.ones(32000, dtype=np.float32)
        with patch("audio_processor.librosa") as mock_lib:
            # Alternating: 5 speech frames, 20 silence frames (20 frames > min_frames at 16kHz)
            rms_pattern = [-5.0] * 5 + [-50.0] * 20 + [-5.0] * 5 + [-50.0] * 20
            mock_lib.feature.rms.return_value = np.array([rms_pattern])
            mock_lib.amplitude_to_db.side_effect = lambda x, ref=None: x
            result = extract_pause_count(y, 16000)
        # 2 silence regions → 2 pauses
        assert result >= 1

    def test_returns_int(self):
        from audio_processor import extract_pause_count
        y = np.zeros(16000, dtype=np.float32)
        with patch("audio_processor.librosa") as mock_lib:
            mock_lib.feature.rms.return_value = np.array([[0.1] * 10])
            mock_lib.amplitude_to_db.return_value = np.array([-5.0] * 10)
            result = extract_pause_count(y, 16000)
        assert isinstance(result, int)


# ── TestConvertToWav ──────────────────────────────────────────────────────────

class TestConvertToWav:

    def test_raises_if_input_not_found(self):
        from audio_processor import convert_to_wav
        with pytest.raises(FileNotFoundError):
            convert_to_wav("/nonexistent/path/audio.webm", "/tmp/out.wav")

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        from audio_processor import convert_to_wav
        # Create a real (but non-audio) input file
        input_file = tmp_path / "test.webm"
        input_file.write_bytes(b"not real audio")

        with patch("audio_processor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error message")
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                convert_to_wav(str(input_file), str(tmp_path / "out.wav"))

    def test_raises_if_ffmpeg_not_found(self, tmp_path):
        from audio_processor import convert_to_wav
        input_file = tmp_path / "test.webm"
        input_file.write_bytes(b"data")

        with patch("audio_processor.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                convert_to_wav(str(input_file), str(tmp_path / "out.wav"))

    def test_calls_ffmpeg_with_correct_args(self, tmp_path):
        from audio_processor import convert_to_wav
        input_file = tmp_path / "test.webm"
        input_file.write_bytes(b"data")
        output_file = str(tmp_path / "out.wav")

        with patch("audio_processor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            convert_to_wav(str(input_file), output_file)

        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args

    def test_succeeds_on_zero_returncode(self, tmp_path):
        from audio_processor import convert_to_wav
        input_file = tmp_path / "test.wav"
        input_file.write_bytes(b"data")

        with patch("audio_processor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Should not raise
            convert_to_wav(str(input_file), str(tmp_path / "out.wav"))


# ── TestTranscribeEndpoint ─────────────────────────────────────────────────────

class TestTranscribeEndpoint:
    """
    Tests for POST /transcribe.
    Mocks transcriber.transcribe() and audio_processor.process_audio_file()
    so no real audio, ffmpeg, or Whisper is needed.
    """

    @pytest.fixture(autouse=True)
    def setup_stubs(self):
        """Stub whisper module and transcriber model before importing main."""
        # Stub whisper
        if "whisper" not in sys.modules:
            w = types.ModuleType("whisper")
            w.load_model = MagicMock(return_value=MagicMock())
            sys.modules["whisper"] = w

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        # Ensure the ASR service dir is at the FRONT of sys.path so 'import main'
        # finds asr_service/main.py and not sentiment_service/main.py
        if ASR_DIR not in sys.path or sys.path[0] != ASR_DIR:
            if ASR_DIR in sys.path:
                sys.path.remove(ASR_DIR)
            sys.path.insert(0, ASR_DIR)

        # Force fresh import of this service's main/transcriber
        for mod in ["main", "transcriber", "audio_processor"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import transcriber as tr
        tr._whisper_model = MagicMock()  # mark as loaded
        tr._model_size    = "small"

        import main as asr_main
        with TestClient(asr_main.app) as c:
            yield c

    def _audio_bytes(self) -> bytes:
        """Minimal fake audio payload."""
        return b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 36

    def test_returns_200_with_mocked_services(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(
            duration_seconds  = 10.0,
            speaking_rate_wpm = 140.0,
            pause_count       = 2,
            pitch_mean_hz     = 185.0,
            pitch_std         = 18.0,
            volume_variation  = 0.12,
            is_fast_speech    = False,
            is_nervous_pitch  = False,
            delivery_flags    = [],
        )
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "Binary search divides the array in half."}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("test.wav", self._audio_bytes(), "audio/wav")},
            )
        assert resp.status_code == 200

    def test_response_schema(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(
            duration_seconds=10.0, speaking_rate_wpm=140.0, pause_count=2,
            pitch_mean_hz=185.0, pitch_std=18.0, volume_variation=0.12,
            is_fast_speech=False, is_nervous_pitch=False, delivery_flags=[],
        )
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "Hello world."}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("test.wav", self._audio_bytes(), "audio/wav")},
            )
        data = resp.json()
        required = [
            "transcript", "speaking_rate_wpm", "pause_count",
            "pitch_mean_hz", "pitch_std", "volume_variation",
            "is_fast_speech", "is_nervous_pitch", "delivery_flags",
        ]
        for key in required:
            assert key in data, f"Missing field: {key}"

    def test_transcript_is_string(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(speaking_rate_wpm=140.0)
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "The answer is recursion."}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        assert isinstance(resp.json()["transcript"], str)

    def test_delivery_flags_is_list(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(
            delivery_flags=["You spoke at 180 WPM — try slowing to 130–150 WPM."]
        )
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "Some answer."}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        flags = resp.json()["delivery_flags"]
        assert isinstance(flags, list)

    def test_too_short_returns_422(self, client):
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "Short."}),
            patch("main.ap.process_audio_file",
                  side_effect=ValueError("Audio too short")),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        assert resp.status_code == 422

    def test_transcription_error_returns_500(self, client):
        with patch("main.transcriber.transcribe",
                   side_effect=RuntimeError("Transcription failed")):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        assert resp.status_code == 500

    def test_missing_audio_file_returns_422(self, client):
        resp = client.post("/transcribe", data={})
        assert resp.status_code == 422

    def test_503_when_model_not_loaded(self, client):
        import transcriber as tr
        original = tr._whisper_model
        tr._whisper_model = None  # simulate model not loaded
        try:
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
            assert resp.status_code == 503
        finally:
            tr._whisper_model = original

    def test_fast_speech_flag_reflected_in_response(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(
            speaking_rate_wpm = 180.0,
            is_fast_speech    = True,
            delivery_flags    = ["You spoke at 180 WPM — try slowing to 130–150 WPM."],
        )
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "answer " * 100}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        data = resp.json()
        assert data["is_fast_speech"] is True
        assert data["speaking_rate_wpm"] > 165

    def test_numeric_fields_are_numbers(self, client):
        from audio_processor import AudioFeatures
        fake_features = AudioFeatures(
            speaking_rate_wpm=140.0, pause_count=2, pitch_mean_hz=185.0,
            pitch_std=18.0, volume_variation=0.12,
        )
        with (
            patch("main.transcriber.transcribe",
                  return_value={"transcript": "Answer text here."}),
            patch("main.ap.process_audio_file", return_value=fake_features),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio_file": ("t.wav", self._audio_bytes(), "audio/wav")},
            )
        data = resp.json()
        assert isinstance(data["speaking_rate_wpm"], (int, float))
        assert isinstance(data["pause_count"],       int)
        assert isinstance(data["pitch_std"],         (int, float))


# ── TestHealthEndpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        if ASR_DIR not in sys.path or sys.path[0] != ASR_DIR:
            if ASR_DIR in sys.path:
                sys.path.remove(ASR_DIR)
            sys.path.insert(0, ASR_DIR)

        if "whisper" not in sys.modules:
            w = types.ModuleType("whisper")
            w.load_model = MagicMock()
            sys.modules["whisper"] = w

        for mod in ["main", "transcriber", "audio_processor"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import transcriber as tr
        tr._whisper_model = MagicMock()
        tr._model_size    = "small"

        import main as asr_main
        with TestClient(asr_main.app) as c:
            yield c

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status"     in data
        assert "model"      in data
        assert "loaded"     in data
        assert "model_size" in data

    def test_health_loaded_true_when_model_ready(self, client):
        data = client.get("/health").json()
        assert data["loaded"] is True
        assert data["status"] == "ok"

    def test_health_model_name(self, client):
        data = client.get("/health").json()
        assert data["model"] == "openai-whisper"

    def test_health_model_size_small(self, client):
        data = client.get("/health").json()
        assert data["model_size"] == "small"
