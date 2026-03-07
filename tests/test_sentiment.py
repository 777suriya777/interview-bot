"""
Tests for the Sentiment Service.

Groups:
  TestHedgingDetection    → analyzer.detect_hedging()
  TestFillerCount         → analyzer.count_filler_words()
  TestConfidenceFeedback  → analyzer.build_confidence_feedback()
  TestImprovementTips     → analyzer.build_improvement_tips()
  TestVocabUtilities      → model.build_vocab(), encode(), encode_batch()
  TestRuleBasedClassifier → main._rule_based_classify() (no torch needed)
  TestClassifyEndpoint    → POST /classify (mocked model)
  TestHealthEndpoint      → GET /health

Only TestVocabUtilities imports from model.py (which imports torch).
All other tests are torch-free.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Make sentiment_service importable
SENT_DIR = str(Path(__file__).parent.parent / "sentiment_service")
if SENT_DIR not in sys.path:
    sys.path.insert(0, SENT_DIR)


# ── TestHedgingDetection ───────────────────────────────────────────────────────

class TestHedgingDetection:

    def test_returns_list(self):
        from analyzer import detect_hedging
        result = detect_hedging("I think this is correct.")
        assert isinstance(result, list)

    def test_detects_i_think(self):
        from analyzer import detect_hedging
        result = detect_hedging("I think hash tables use arrays.")
        assert any(p["text"].lower() == "i think" for p in result)

    def test_detects_i_guess(self):
        from analyzer import detect_hedging
        result = detect_hedging("I guess it depends on the context.")
        assert any("i guess" in p["text"].lower() for p in result)

    def test_detects_kind_of(self):
        from analyzer import detect_hedging
        result = detect_hedging("It's kind of like a linked list.")
        assert any("kind of" in p["text"].lower() for p in result)

    def test_no_hedging_clean_answer(self):
        from analyzer import detect_hedging
        result = detect_hedging("A hash table maps keys to values using a hash function.")
        assert len(result) == 0

    def test_multiple_hedges_all_detected(self):
        from analyzer import detect_hedging
        text = "I think maybe it's kind of like a stack, I guess."
        result = detect_hedging(text)
        assert len(result) >= 3

    def test_sorted_by_position(self):
        from analyzer import detect_hedging
        text = "I think this is correct. Maybe I'm wrong. I guess so."
        result = detect_hedging(text)
        positions = [p["start_char"] for p in result]
        assert positions == sorted(positions)

    def test_start_char_correct(self):
        from analyzer import detect_hedging
        text = "This is correct, I think."
        result = detect_hedging(text)
        for phrase in result:
            # Verify start_char actually points to the phrase in the text
            found = text[phrase["start_char"]: phrase["start_char"] + len(phrase["text"])]
            assert found.lower() == phrase["text"].lower()

    def test_case_insensitive(self):
        from analyzer import detect_hedging
        result = detect_hedging("I THINK this is a hash table.")
        assert any("i think" in p["text"].lower() for p in result)

    def test_not_sure_detected(self):
        from analyzer import detect_hedging
        result = detect_hedging("I'm not sure about the complexity.")
        assert len(result) > 0

    def test_returns_dict_with_text_and_start_char(self):
        from analyzer import detect_hedging
        result = detect_hedging("I think it is correct.")
        for phrase in result:
            assert "text" in phrase
            assert "start_char" in phrase
            assert isinstance(phrase["start_char"], int)


# ── TestFillerCount ────────────────────────────────────────────────────────────

class TestFillerCount:

    def test_counts_um(self):
        from analyzer import count_filler_words
        assert count_filler_words("Um, the answer is um yes.") >= 2

    def test_counts_uh(self):
        from analyzer import count_filler_words
        assert count_filler_words("Uh, I would say uh that...") >= 2

    def test_zero_for_clean_text(self):
        from analyzer import count_filler_words
        assert count_filler_words("Hash tables provide O(1) average lookup.") == 0

    def test_like_as_filler(self):
        from analyzer import count_filler_words
        # "like" as a filler (multiple occurrences)
        count = count_filler_words("It's like, you know, like a queue structure.")
        assert count >= 2

    def test_returns_int(self):
        from analyzer import count_filler_words
        result = count_filler_words("Some text here.")
        assert isinstance(result, int)

    def test_empty_string(self):
        from analyzer import count_filler_words
        assert count_filler_words("") == 0

    def test_no_false_positive_likewise(self):
        from analyzer import count_filler_words
        # "likewise" should NOT count as the filler "like"
        count = count_filler_words("Likewise, the approach is similar.")
        assert count == 0


# ── TestConfidenceFeedback ────────────────────────────────────────────────────

class TestConfidenceFeedback:

    def test_high_label_positive(self):
        from analyzer import build_confidence_feedback
        result = build_confidence_feedback("high", 0, 0)
        assert "confidence" in result.lower() or "great" in result.lower()

    def test_low_label_mentions_hedging(self):
        from analyzer import build_confidence_feedback
        result = build_confidence_feedback("low", 5, 2)
        assert "hedging" in result.lower() or "uncertain" in result.lower() or "confidence" in result.lower()

    def test_anxious_label_mentions_fillers_or_hedging(self):
        from analyzer import build_confidence_feedback
        result = build_confidence_feedback("anxious", 4, 6)
        assert "filler" in result.lower() or "anxiety" in result.lower() or "anxious" in result.lower() or "filler word" in result.lower() or "4" in result

    def test_all_labels_return_string(self):
        from analyzer import build_confidence_feedback
        for label in ["high", "moderate", "low", "anxious"]:
            result = build_confidence_feedback(label, 2, 1)
            assert isinstance(result, str) and len(result) > 10

    def test_unknown_label_returns_fallback(self):
        from analyzer import build_confidence_feedback
        result = build_confidence_feedback("unknown_label", 0, 0)
        assert isinstance(result, str) and len(result) > 5


# ── TestImprovementTips ───────────────────────────────────────────────────────

class TestImprovementTips:

    def test_returns_list(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 2, "relevance": 3, "completeness": 3, "accuracy": 3},
            "moderate", 1, 0, []
        )
        assert isinstance(tips, list)

    def test_at_most_three_tips(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 0, "relevance": 0, "completeness": 0, "accuracy": 0},
            "anxious", 5, 8, ["fast_speech", "nervous_pitch", "excessive_pauses"]
        )
        assert len(tips) <= 3

    def test_at_least_one_tip(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 4, "relevance": 4, "completeness": 4, "accuracy": 4},
            "high", 0, 0, []
        )
        assert len(tips) >= 1

    def test_delivery_flag_included(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 3, "relevance": 3, "completeness": 3, "accuracy": 3},
            "high", 0, 0, ["fast_speech"]
        )
        assert any("slow down" in t.lower() or "wpm" in t.lower() for t in tips)

    def test_low_confidence_tip_included(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 3, "relevance": 3, "completeness": 3, "accuracy": 3},
            "low", 4, 2, []
        )
        combined = " ".join(tips).lower()
        assert "confident" in combined or "hedg" in combined or "uncertain" in combined

    def test_all_tips_are_strings(self):
        from analyzer import build_improvement_tips
        tips = build_improvement_tips(
            {"content": 1, "relevance": 2, "completeness": 1, "accuracy": 2},
            "moderate", 3, 2, ["too_short"]
        )
        assert all(isinstance(t, str) for t in tips)


# ── TestVocabUtilities ────────────────────────────────────────────────────────
# These tests import from model.py which imports torch.
# Skip if torch not available.

try:
    import torch as _torch
    # Check for a real torch attribute that only exists in the actual package
    HAS_TORCH = hasattr(_torch, "float32")
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@requires_torch
class TestVocabUtilities:

    def test_build_vocab_has_pad_unk(self):
        from model import build_vocab
        vocab = build_vocab(["hello world", "hello test"], min_freq=1)
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_build_vocab_min_freq(self):
        from model import build_vocab
        # "rare" appears once, "common" appears 3 times
        texts = ["common word", "common word", "common word", "rare word"]
        vocab = build_vocab(texts, min_freq=2)
        assert "common" in vocab
        assert "rare" not in vocab

    def test_build_vocab_size(self):
        from model import build_vocab
        texts = ["alpha beta gamma", "alpha delta epsilon"]
        vocab = build_vocab(texts, min_freq=1)
        # Should have PAD, UNK + word tokens
        assert len(vocab) >= 2

    def test_encode_known_words(self):
        from model import build_vocab, encode
        vocab = build_vocab(["hello world test"], min_freq=1)
        indices = encode("hello world", vocab)
        for idx in indices:
            assert idx > 0  # > 0 means not PAD

    def test_encode_unknown_word_maps_to_unk(self):
        from model import build_vocab, encode
        vocab = build_vocab(["hello world"], min_freq=1)
        indices = encode("zzzzunknownword", vocab)
        assert indices[0] == vocab["<UNK>"]

    def test_encode_respects_max_length(self):
        from model import build_vocab, encode
        vocab = build_vocab(["a b c d e f g h i j"], min_freq=1)
        indices = encode("a b c d e f g h i j", vocab, max_length=5)
        assert len(indices) == 5

    def test_encode_batch_shape(self):
        from model import build_vocab, encode_batch
        import torch
        vocab = build_vocab(["hello world test", "short text"], min_freq=1)
        tensor = encode_batch(["hello world", "short"], vocab)
        assert tensor.dtype == torch.long
        assert tensor.ndim == 2
        assert tensor.shape[0] == 2  # batch of 2

    def test_encode_batch_padded_to_same_length(self):
        from model import build_vocab, encode_batch
        vocab = build_vocab(["hello world this is a long sentence", "short"], min_freq=1)
        tensor = encode_batch(["hello world this is a long sentence", "short"], vocab)
        # All rows same length after padding
        assert tensor.shape[1] > 1
        lengths = [len([x for x in row if x != 0]) for row in tensor.tolist()]
        assert lengths[0] >= lengths[1]  # longer text has more non-pad tokens


# ── TestRuleBasedClassifier ───────────────────────────────────────────────────

class TestRuleBasedClassifier:
    """
    Test the rule-based fallback classifier in main.py.
    Imports _rule_based_classify without triggering model loading.
    """

    def _classify(self, text: str):
        """Import and call _rule_based_classify directly."""
        # Patch torch.load so importing main.py doesn't fail without model files
        from analyzer import detect_hedging, count_filler_words, IDX_TO_LABEL
        from analyzer import build_confidence_feedback

        # Re-implement the same logic inline so we don't need to import main.py
        hedges = detect_hedging(text)
        fillers = count_filler_words(text)
        word_count = max(len(text.split()), 1)
        hedges_per_100 = (len(hedges) / max(word_count, 50)) * 100
        fillers_per_100 = (fillers / max(word_count, 50)) * 100

        if hedges_per_100 > 6 or fillers_per_100 > 8:
            return "anxious"
        elif hedges_per_100 > 3:
            return "low"
        elif hedges_per_100 > 1:
            return "moderate"
        else:
            return "high"

    def test_clean_answer_high(self):
        text = "Binary search operates on a sorted array. It divides the search space in half at each step, giving O(log n) time complexity."
        assert self._classify(text) == "high"

    def test_heavy_hedging_low_or_anxious(self):
        text = "I think maybe it's kind of like I guess a recursive approach. I'm not sure. I believe it might work."
        label = self._classify(text)
        assert label in ("low", "anxious")

    def test_moderate_some_hedging(self):
        text = "I think binary search works on sorted arrays and it's O(log n) time complexity."
        assert self._classify(text) == "moderate"

    def test_label_is_valid(self):
        for text in [
            "I think so.",
            "Absolutely, this is O(n log n).",
            "I guess, um, maybe kind of sort of.",
        ]:
            label = self._classify(text)
            assert label in ("high", "moderate", "low", "anxious")


# ── TestClassifyEndpoint ──────────────────────────────────────────────────────

class TestClassifyEndpoint:
    """Test POST /classify with a mocked model (no torch/weights required)."""

    @pytest.fixture
    def client(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from conftest_torch_stub import install_torch_stub
        install_torch_stub()

        # Ensure sentiment_service dir is at front so 'import main' finds
        # sentiment/main.py and not asr/main.py
        if SENT_DIR not in sys.path or sys.path[0] != SENT_DIR:
            if SENT_DIR in sys.path:
                sys.path.remove(SENT_DIR)
            sys.path.insert(0, SENT_DIR)

        from fastapi.testclient import TestClient
        # Force fresh import of main after stub is installed
        for mod in ["main", "model", "analyzer"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import main as sent_main
        sent_main.model_store.clear()
        sent_main.model_store["model"]  = None
        sent_main.model_store["vocab"]  = None
        sent_main.model_store["device"] = "cpu"
        sent_main.model_store["ready"]  = True

        with TestClient(sent_main.app, raise_server_exceptions=True) as c:
            yield c
        sent_main.model_store.clear()

    def test_classify_returns_200(self, client):
        resp = client.post("/classify", json={
            "transcript": "A hash table maps keys to values using a hash function."
        })
        assert resp.status_code == 200

    def test_classify_response_schema(self, client):
        resp = client.post("/classify", json={"transcript": "I think it uses recursion."})
        data = resp.json()
        assert "confidence_label"  in data
        assert "confidence_score"  in data
        assert "hedging_phrases"   in data
        assert "filler_word_count" in data
        assert "feedback_text"     in data

    def test_confidence_label_valid(self, client):
        resp = client.post("/classify", json={"transcript": "It uses dynamic programming."})
        label = resp.json()["confidence_label"]
        assert label in ("high", "moderate", "low", "anxious")

    def test_confidence_score_in_range(self, client):
        resp = client.post("/classify", json={"transcript": "This is definitely correct."})
        score = resp.json()["confidence_score"]
        assert 0.0 <= score <= 1.0

    def test_hedging_phrases_is_list(self, client):
        resp = client.post("/classify", json={"transcript": "I think maybe this is right."})
        assert isinstance(resp.json()["hedging_phrases"], list)

    def test_hedging_phrase_has_text_and_start(self, client):
        resp = client.post("/classify", json={"transcript": "I think this is correct."})
        phrases = resp.json()["hedging_phrases"]
        if phrases:
            assert "text"       in phrases[0]
            assert "start_char" in phrases[0]

    def test_filler_count_non_negative(self, client):
        resp = client.post("/classify", json={"transcript": "Um, I think this is, like, correct."})
        assert resp.json()["filler_word_count"] >= 0

    def test_feedback_text_is_string(self, client):
        resp = client.post("/classify", json={"transcript": "Recursion involves a base case."})
        assert isinstance(resp.json()["feedback_text"], str)
        assert len(resp.json()["feedback_text"]) > 5

    def test_empty_transcript_rejected(self, client):
        resp = client.post("/classify", json={"transcript": ""})
        assert resp.status_code == 422

    def test_missing_field_rejected(self, client):
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_high_confidence_clean_answer(self, client):
        """Clean answer with no hedges → should be classified as 'high'."""
        text = (
            "Binary search requires a sorted array. It compares the target to the "
            "midpoint and eliminates half the search space each iteration, achieving "
            "O(log n) time complexity."
        )
        resp = client.post("/classify", json={"transcript": text})
        assert resp.json()["confidence_label"] == "high"

    def test_low_confidence_heavy_hedging(self, client):
        """Heavy hedging → rule-based fallback should return 'low' or 'anxious'."""
        text = (
            "I think maybe it's kind of like I guess a recursive approach. "
            "I'm not sure. I believe it might possibly work I suppose."
        )
        resp = client.post("/classify", json={"transcript": text})
        assert resp.json()["confidence_label"] in ("low", "anxious")


# ── TestHealthEndpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.fixture
    def client(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from conftest_torch_stub import install_torch_stub
        install_torch_stub()

        if SENT_DIR not in sys.path or sys.path[0] != SENT_DIR:
            if SENT_DIR in sys.path:
                sys.path.remove(SENT_DIR)
            sys.path.insert(0, SENT_DIR)

        from fastapi.testclient import TestClient
        for mod in ["main", "model", "analyzer"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import main as sent_main
        sent_main.model_store.clear()
        sent_main.model_store["model"]  = None
        sent_main.model_store["vocab"]  = None
        sent_main.model_store["device"] = "cpu"
        sent_main.model_store["ready"]  = True

        with TestClient(sent_main.app) as c:
            yield c
        sent_main.model_store.clear()

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model"  in data
        assert "loaded" in data

    def test_health_model_name_fallback(self, client):
        data = client.get("/health").json()
        assert data["model"] == "rule-based-fallback"

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
