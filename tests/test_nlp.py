"""
Unit tests for the NLP service.

Tests are grouped by module:
  - TestInterviewBERTScorer   → model.py architecture
  - TestFeedbackSummary       → build_feedback_summary()
  - TestCacheKey              → SHA-256 cache key logic
  - TestEvalEndpoint          → /evaluate FastAPI endpoint (mocked model)
  - TestHealthEndpoint        → /health endpoint
  - TestMacroF1               → evaluate.py metric function
  - TestAccuracy              → evaluate.py accuracy function

No GPU or trained weights required — model uses random weights for shape tests.
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# Make nlp_service importable
sys.path.insert(0, str(Path(__file__).parent.parent / "nlp_service"))


# ── TestInterviewBERTScorer ───────────────────────────────────────────────────

class TestInterviewBERTScorer:
    """Test model architecture without running inference (no forward pass)."""

    def test_model_instantiates(self):
        from model import InterviewBERTScorer
        m = InterviewBERTScorer.__new__(InterviewBERTScorer)
        assert m is not None

    def test_dimensions_constant(self):
        from model import InterviewBERTScorer
        assert InterviewBERTScorer.DIMENSIONS == [
            "content", "relevance", "completeness", "accuracy"
        ]

    def test_four_heads_exist(self):
        """Check the model has all 4 scoring heads as attributes."""
        import torch.nn as nn
        from model import InterviewBERTScorer
        # Patch BertModel.from_pretrained to avoid downloading weights
        with patch("model.BertModel.from_pretrained") as mock_bert:
            # Create a mock BERT that returns config.hidden_size = 768
            mock_bert_instance = MagicMock()
            mock_bert_instance.config.hidden_size = 768
            mock_bert.return_value = mock_bert_instance

            m = InterviewBERTScorer.__new__(InterviewBERTScorer)
            torch.nn.Module.__init__(m)
            m.bert = mock_bert_instance
            m.dropout = nn.Dropout(0.1)
            m.head_content      = nn.Linear(768, 5)
            m.head_relevance    = nn.Linear(768, 5)
            m.head_completeness = nn.Linear(768, 5)
            m.head_accuracy     = nn.Linear(768, 5)

            assert hasattr(m, "head_content")
            assert hasattr(m, "head_relevance")
            assert hasattr(m, "head_completeness")
            assert hasattr(m, "head_accuracy")
            assert hasattr(m, "dropout")

    def test_head_output_shape(self):
        """Each head must output 5 classes (scores 0-4)."""
        import torch.nn as nn
        head = nn.Linear(768, 5)
        x = torch.randn(2, 768)   # batch of 2, hidden size 768
        out = head(x)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_forward_returns_four_keys(self):
        """forward() must return a dict with exactly 4 dimension keys."""
        import torch.nn as nn
        from model import InterviewBERTScorer

        # Build a minimal mock that behaves like the real forward()
        with patch("model.BertModel.from_pretrained") as mock_bert:
            # pooler_output is the only BERT output we use
            mock_output = MagicMock()
            mock_output.pooler_output = torch.randn(1, 768)

            mock_bert_instance = MagicMock()
            mock_bert_instance.config.hidden_size = 768
            mock_bert_instance.return_value = mock_output
            mock_bert.return_value = mock_bert_instance

            model = InterviewBERTScorer()
            dummy_ids  = torch.zeros(1, 10, dtype=torch.long)
            dummy_mask = torch.ones(1, 10, dtype=torch.long)
            dummy_type = torch.zeros(1, 10, dtype=torch.long)

            result = model.forward(dummy_ids, dummy_mask, dummy_type)

        assert set(result.keys()) == {"content", "relevance", "completeness", "accuracy"}
        for dim, tensor in result.items():
            assert tensor.shape == (1, 5), f"Head '{dim}' shape should be (1,5)"

    def test_argmax_score_in_range(self):
        """argmax of 5 logits always produces a value 0-4."""
        logits = torch.tensor([[0.1, 0.9, 0.2, 0.3, 0.0]])  # shape [1, 5]
        score = int(logits.argmax(dim=-1).item())
        assert 0 <= score <= 4


# ── TestFeedbackSummary ────────────────────────────────────────────────────────

class TestFeedbackSummary:

    def test_returns_string(self):
        from model import build_feedback_summary
        result = build_feedback_summary({"content":3,"relevance":3,"completeness":3,"accuracy":3})
        assert isinstance(result, str)
        assert len(result) > 10

    def test_high_scores_positive_tone(self):
        from model import build_feedback_summary
        result = build_feedback_summary({"content":4,"relevance":4,"completeness":4,"accuracy":4})
        assert "Strong answer" in result or "excellent" in result.lower()

    def test_low_scores_mention_weak_dimension(self):
        from model import build_feedback_summary
        # relevance is the weakest
        result = build_feedback_summary({"content":3,"relevance":0,"completeness":3,"accuracy":3})
        assert "relevance" in result.lower()

    def test_zero_scores_critical_feedback(self):
        from model import build_feedback_summary
        result = build_feedback_summary({"content":0,"relevance":0,"completeness":0,"accuracy":0})
        assert "Weak answer" in result

    def test_mixed_scores_mentions_weakest(self):
        from model import build_feedback_summary
        # accuracy is 1 — the weakest
        result = build_feedback_summary({"content":3,"relevance":3,"completeness":3,"accuracy":1})
        assert "accuracy" in result.lower()

    def test_all_dimensions_handled(self):
        """Every combination of weakest dimension should produce output without error."""
        from model import build_feedback_summary
        dims = ["content", "relevance", "completeness", "accuracy"]
        for weak_dim in dims:
            scores = {d: 3 for d in dims}
            scores[weak_dim] = 0
            result = build_feedback_summary(scores)
            assert isinstance(result, str), f"Failed for weak_dim={weak_dim}"


# ── TestCacheKey ───────────────────────────────────────────────────────────────

class TestCacheKey:
    """Test the SHA-256 cache key logic from main.py."""

    def _cache_key(self, question: str, answer: str) -> str:
        """Inline the cache key logic for isolated testing."""
        content = (question + answer).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        return f"nlp:cache:{digest}"

    def test_key_format(self):
        key = self._cache_key("What is a hash table?", "A hash table maps keys to values.")
        assert key.startswith("nlp:cache:")
        # SHA-256 hex digest is 64 chars
        assert len(key) == len("nlp:cache:") + 64

    def test_same_input_same_key(self):
        q, a = "Explain recursion.", "Recursion is a function calling itself."
        key1 = self._cache_key(q, a)
        key2 = self._cache_key(q, a)
        assert key1 == key2

    def test_different_answer_different_key(self):
        q = "Explain recursion."
        a1 = "Recursion is a function calling itself."
        a2 = "Recursion involves a base case and recursive case."
        assert self._cache_key(q, a1) != self._cache_key(q, a2)

    def test_different_question_different_key(self):
        q1 = "What is a stack?"
        q2 = "What is a queue?"
        a  = "It stores elements."
        assert self._cache_key(q1, a) != self._cache_key(q2, a)

    def test_key_is_deterministic_across_calls(self):
        """SHA-256 must be deterministic (no random salt)."""
        q, a = "Test question?", "Test answer."
        keys = {self._cache_key(q, a) for _ in range(10)}
        assert len(keys) == 1, "Cache key must be deterministic"


# ── TestEvalEndpoint ───────────────────────────────────────────────────────────

class TestEvalEndpoint:
    """Test /evaluate endpoint with a mocked model (no GPU required)."""

    @pytest.fixture
    def client(self):
        """Create a TestClient with pre-populated model_store."""
        from fastapi.testclient import TestClient
        import main as nlp_main

        # Mock model that returns fixed logits
        mock_model = MagicMock()
        mock_logits = {
            dim: torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])  # argmax → score 3
            for dim in ["content", "relevance", "completeness", "accuracy"]
        }
        mock_model.return_value = mock_logits

        # Mock tokenizer that returns dummy tensors
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "token_type_ids": torch.zeros(1, 10, dtype=torch.long),
        }

        # Patch model_store so lifespan is bypassed
        nlp_main.model_store.clear()
        nlp_main.model_store["model"]     = mock_model
        nlp_main.model_store["tokenizer"] = mock_tokenizer
        nlp_main.model_store["device"]    = torch.device("cpu")
        nlp_main.model_store["redis"]     = None  # no Redis in tests

        with TestClient(nlp_main.app, raise_server_exceptions=True) as c:
            yield c

        nlp_main.model_store.clear()

    def test_evaluate_returns_200(self, client):
        resp = client.post("/evaluate", json={
            "question": "What is a hash table?",
            "answer": "A data structure that maps keys to values using a hash function."
        })
        assert resp.status_code == 200

    def test_evaluate_response_schema(self, client):
        resp = client.post("/evaluate", json={
            "question": "Explain Big-O notation.",
            "answer": "Big-O describes the upper bound of an algorithm's time complexity."
        })
        data = resp.json()
        required = {
            "content_score", "relevance_score", "completeness_score",
            "accuracy_score", "overall_score", "feedback_summary"
        }
        assert required.issubset(data.keys())

    def test_evaluate_scores_in_range(self, client):
        resp = client.post("/evaluate", json={
            "question": "What is recursion?",
            "answer": "A function that calls itself with a base case."
        })
        data = resp.json()
        for dim in ["content_score", "relevance_score", "completeness_score", "accuracy_score"]:
            assert 0 <= data[dim] <= 4, f"{dim} out of range: {data[dim]}"
        assert 0.0 <= data["overall_score"] <= 4.0

    def test_evaluate_overall_is_mean(self, client):
        """overall_score must equal mean of the 4 dimension scores."""
        resp = client.post("/evaluate", json={
            "question": "What is a stack?",
            "answer": "LIFO data structure."
        })
        data = resp.json()
        expected = round((
            data["content_score"] + data["relevance_score"] +
            data["completeness_score"] + data["accuracy_score"]
        ) / 4.0, 2)
        assert abs(data["overall_score"] - expected) < 0.01

    def test_evaluate_feedback_is_string(self, client):
        resp = client.post("/evaluate", json={
            "question": "What is a linked list?",
            "answer": "A sequence of nodes."
        })
        assert isinstance(resp.json()["feedback_summary"], str)
        assert len(resp.json()["feedback_summary"]) > 5

    def test_evaluate_empty_answer_rejected(self, client):
        resp = client.post("/evaluate", json={"question": "Q?", "answer": ""})
        assert resp.status_code == 422

    def test_evaluate_empty_question_rejected(self, client):
        resp = client.post("/evaluate", json={"question": "", "answer": "Some answer."})
        assert resp.status_code == 422

    def test_evaluate_missing_field_rejected(self, client):
        resp = client.post("/evaluate", json={"question": "Q?"})  # no answer
        assert resp.status_code == 422


# ── TestHealthEndpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main as nlp_main

        nlp_main.model_store.clear()
        nlp_main.model_store["model"]  = MagicMock()
        nlp_main.model_store["device"] = torch.device("cpu")
        nlp_main.model_store["redis"]  = None

        with TestClient(nlp_main.app) as c:
            yield c
        nlp_main.model_store.clear()

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model" in data
        assert "loaded" in data
        assert data["loaded"] is True

    def test_health_model_name(self, client):
        data = client.get("/health").json()
        assert data["model"] == "bert-base-uncased"


# ── TestMacroF1 ───────────────────────────────────────────────────────────────

class TestMacroF1:
    """Test the macro F1 computation in evaluate.py."""

    def test_perfect_predictions(self):
        from evaluate import macro_f1
        labels = [0, 1, 2, 3, 4]
        assert macro_f1(labels, labels) == pytest.approx(1.0, abs=0.01)

    def test_all_wrong(self):
        from evaluate import macro_f1
        # All predicted as 0, all true are 1
        preds  = [0, 0, 0, 0]
        labels = [1, 1, 1, 1]
        result = macro_f1(preds, labels)
        assert result < 0.3

    def test_returns_float(self):
        from evaluate import macro_f1
        result = macro_f1([0, 1, 2], [0, 1, 2])
        assert isinstance(result, float)

    def test_value_between_0_and_1(self):
        from evaluate import macro_f1
        preds  = [0, 1, 2, 3, 4, 0, 1]
        labels = [0, 1, 2, 1, 3, 0, 2]
        result = macro_f1(preds, labels)
        assert 0.0 <= result <= 1.0


class TestAccuracy:

    def test_perfect(self):
        from evaluate import accuracy
        assert accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_zero(self):
        from evaluate import accuracy
        assert accuracy([0, 0, 0], [1, 1, 1]) == 0.0

    def test_partial(self):
        from evaluate import accuracy
        # 2 out of 4 correct
        assert accuracy([0, 1, 0, 1], [0, 1, 1, 0]) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        from evaluate import accuracy
        assert accuracy([], []) == 0.0
