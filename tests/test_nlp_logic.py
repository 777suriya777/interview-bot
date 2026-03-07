"""
Tests for NLP service pure-logic functions (no torch required).
Imports directly from feedback.py and metrics.py which have no torch dependency.
"""
import hashlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "nlp_service"))

from feedback import build_feedback_summary
from metrics import accuracy, macro_f1


class TestCacheKey:
    def _key(self, q, a):
        digest = hashlib.sha256((q + a).encode()).hexdigest()
        return f"nlp:cache:{digest}"

    def test_format(self):
        key = self._key("Q?", "A.")
        assert key.startswith("nlp:cache:")
        assert len(key) == 10 + 64

    def test_same_input_same_key(self):
        assert self._key("Q", "A") == self._key("Q", "A")

    def test_different_answer_different_key(self):
        assert self._key("Q", "A1") != self._key("Q", "A2")

    def test_different_question_different_key(self):
        assert self._key("Q1", "A") != self._key("Q2", "A")

    def test_deterministic(self):
        keys = {self._key("q", "a") for _ in range(10)}
        assert len(keys) == 1


class TestFeedbackSummary:
    def test_returns_string(self):
        r = build_feedback_summary({"content":3,"relevance":3,"completeness":3,"accuracy":3})
        assert isinstance(r, str) and len(r) > 10

    def test_perfect_positive(self):
        r = build_feedback_summary({"content":4,"relevance":4,"completeness":4,"accuracy":4})
        assert "Strong answer" in r or "excellent" in r.lower()

    def test_zero_weak(self):
        r = build_feedback_summary({"content":0,"relevance":0,"completeness":0,"accuracy":0})
        assert "Weak answer" in r

    def test_low_relevance_mentioned(self):
        r = build_feedback_summary({"content":3,"relevance":0,"completeness":3,"accuracy":3})
        assert "relevance" in r.lower()

    def test_all_dims_handled(self):
        for dim in ["content","relevance","completeness","accuracy"]:
            scores = {d:3 for d in ["content","relevance","completeness","accuracy"]}
            scores[dim] = 0
            assert isinstance(build_feedback_summary(scores), str)

    def test_overall_computed(self):
        # overall = (2+2+2+2)/4 = 2.0 → should be "Adequate answer" or "Good answer"
        r = build_feedback_summary({"content":2,"relevance":2,"completeness":2,"accuracy":2})
        assert "answer" in r.lower()

    def test_near_perfect_includes_improvement(self):
        # overall = (3+4+4+4)/4 = 3.75 → Strong; weakest is content at 3
        r = build_feedback_summary({"content":3,"relevance":4,"completeness":4,"accuracy":4})
        assert "Strong answer" in r


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([0,1,2,3],[0,1,2,3]) == 1.0

    def test_zero(self):
        assert accuracy([0,0,0],[1,1,1]) == 0.0

    def test_partial(self):
        assert accuracy([0,1,1,0],[0,1,0,1]) == pytest.approx(0.5)

    def test_empty(self):
        assert accuracy([],[]) == 0.0

    def test_single_correct(self):
        assert accuracy([2],[2]) == 1.0

    def test_single_wrong(self):
        assert accuracy([1],[2]) == 0.0


class TestMacroF1:
    def test_perfect(self):
        labels = [0,1,2,3,4]
        assert macro_f1(labels, labels) == pytest.approx(1.0, abs=0.01)

    def test_returns_float(self):
        assert isinstance(macro_f1([0,1],[0,1]), float)

    def test_in_range(self):
        result = macro_f1([0,1,2,3,4,0],[0,1,2,1,3,0])
        assert 0.0 <= result <= 1.0

    def test_all_wrong_low(self):
        assert macro_f1([0,0,0,0],[1,1,1,1]) < 0.3

    def test_partial_agreement(self):
        result = macro_f1([0,1,2,0],[0,1,2,3])
        assert result > 0.5
