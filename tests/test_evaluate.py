"""
test_evaluate.py — Unit tests for the three evaluation scripts.

Tests cover:
  - WER computation correctness (TestWERComputation)
  - Synthetic data generation statistics (TestSyntheticDataGenerators)
  - Metric functions: accuracy, F1 (TestMetricFunctions)
  - CLI --synthetic smoke-test per script (TestEvaluationCLI)
  - Paper target pass/fail verdicts (TestPaperTargets)
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest

ROOT         = Path(__file__).parent.parent
NLP_DIR      = str(ROOT / "nlp_service")
SENTIMENT_DIR= str(ROOT / "sentiment_service")
ASR_DIR      = str(ROOT / "asr_service")


def _ensure(d: str) -> None:
    if d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)


# ─────────────────────────────────────────────────────────────────────────────
# TestWERComputation
# ─────────────────────────────────────────────────────────────────────────────

class TestWERComputation:
    """evaluate_asr.compute_wer — correctness on known cases."""

    @pytest.fixture(autouse=True)
    def load(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as m
        self.m = m

    def test_identical_strings_zero_wer(self):
        wer, n = self.m.compute_wer("hello world", "hello world")
        assert wer == pytest.approx(0.0)
        assert n   == 2

    def test_one_substitution(self):
        # "hello world" vs "hello earth" → 1 sub / 2 ref = 0.5
        wer, n = self.m.compute_wer("hello world", "hello earth")
        assert wer == pytest.approx(0.5)
        assert n   == 2

    def test_one_deletion(self):
        # ref="a b c", hyp="a c" → 1 deletion / 3 ref = 0.333
        wer, n = self.m.compute_wer("a b c", "a c")
        assert wer == pytest.approx(1/3, rel=0.01)

    def test_one_insertion(self):
        # ref="a b", hyp="a x b" → 1 insertion / 2 ref = 0.5
        wer, n = self.m.compute_wer("a b", "a x b")
        assert wer == pytest.approx(0.5)

    def test_completely_wrong(self):
        # ref="a b c", hyp="x y z" → 3 subs / 3 ref = 1.0
        wer, _ = self.m.compute_wer("a b c", "x y z")
        assert wer == pytest.approx(1.0)

    def test_empty_hypothesis_gives_100pct(self):
        # All ref words deleted
        wer, n = self.m.compute_wer("hello world", "")
        assert wer == pytest.approx(1.0)
        assert n   == 2

    def test_empty_reference_returns_zero(self):
        wer, n = self.m.compute_wer("", "something")
        assert wer == pytest.approx(0.0)
        assert n   == 0

    def test_case_insensitive(self):
        wer, _ = self.m.compute_wer("Hello World", "hello world")
        assert wer == pytest.approx(0.0)

    def test_punctuation_stripped(self):
        wer, _ = self.m.compute_wer("hello, world!", "hello world")
        assert wer == pytest.approx(0.0)

    def test_wer_accumulator_micro(self):
        acc = self.m.WERAccumulator("test")
        acc.add("a b c", "a b c")   # 0 errors / 3 words
        acc.add("a b c", "a x c")   # 1 error  / 3 words
        # micro: 1 total op / 6 total ref words = 0.1667
        assert acc.micro_wer == pytest.approx(1/6, rel=0.01)
        assert acc.utterances == 2

    def test_wer_accumulator_macro(self):
        acc = self.m.WERAccumulator("test")
        acc.add("a b", "a b")    # WER=0.0
        acc.add("a b", "x y")    # WER=1.0
        # macro: mean([0.0, 1.0]) = 0.5
        assert acc.macro_wer == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestSyntheticDataGenerators
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticDataGenerators:
    """Verify each synthetic generator produces data matching paper targets."""

    def test_nlp_synthetic_accuracy_near_target(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        dim_data = ev.generate_synthetic_predictions(n=1000, target_accuracy=0.950, seed=42)
        for dim in ev.DIMS:
            preds, labels = dim_data[dim]
            acc = ev.accuracy_score(preds, labels)
            assert 0.92 <= acc <= 1.0, f"Dim {dim}: accuracy {acc:.3f} out of expected range"

    def test_nlp_synthetic_f1_clears_94pct(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        dim_data = ev.generate_synthetic_predictions(n=1000, target_accuracy=0.950, seed=42)
        f1s = [ev.macro_f1_score(*dim_data[d]) for d in ev.DIMS]
        macro_f1 = sum(f1s) / len(f1s)
        assert macro_f1 >= 0.94, f"Macro F1 {macro_f1:.3f} below 0.94 target"

    def test_nlp_synthetic_returns_correct_keys(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        dim_data = ev.generate_synthetic_predictions(n=100, target_accuracy=0.942, seed=1)
        assert set(dim_data.keys()) == set(ev.DIMS)

    def test_nlp_synthetic_n_samples(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        n = 250
        dim_data = ev.generate_synthetic_predictions(n=n, target_accuracy=0.942, seed=1)
        for dim in ev.DIMS:
            preds, labels = dim_data[dim]
            assert len(preds)  == n
            assert len(labels) == n

    def test_nlp_scores_in_valid_range(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        dim_data = ev.generate_synthetic_predictions(n=100, target_accuracy=0.942, seed=1)
        for dim in ev.DIMS:
            preds, labels = dim_data[dim]
            assert all(0 <= p <= 4 for p in preds)
            assert all(0 <= l <= 4 for l in labels)

    def test_sentiment_synthetic_accuracy_near_target(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        preds, labels = ev.generate_synthetic_predictions(n=2000, target_accuracy=0.918, seed=42)
        acc = ev.accuracy_score(preds, labels)
        assert 0.90 <= acc <= 1.0, f"Accuracy {acc:.3f} out of expected range"

    def test_sentiment_labels_are_valid_class_indices(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        preds, labels = ev.generate_synthetic_predictions(n=500, target_accuracy=0.918, seed=1)
        assert all(0 <= p < ev.NUM_CLASSES for p in preds)
        assert all(0 <= l < ev.NUM_CLASSES for l in labels)

    def test_sentiment_class_distribution_realistic(self):
        """moderate (idx=1) should be most common class in labels."""
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        _, labels = ev.generate_synthetic_predictions(n=5000, target_accuracy=0.918, seed=42)
        counts = [labels.count(i) for i in range(ev.NUM_CLASSES)]
        # moderate is index 1 (weight 35%)
        assert counts[1] == max(counts), "moderate should be the most common class"

    def test_asr_synthetic_wer_clear_below_4pct(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=200, seed=42)
        clear   = [s for s in samples if s.audio_type == "clear"]
        acc     = ev.WERAccumulator("clear")
        for s in clear:
            acc.add(s.reference, s.hypothesis)
        assert acc.micro_wer * 100 <= 4.0, f"Clear WER {acc.micro_wer*100:.2f}% exceeds 4%"

    def test_asr_synthetic_overall_wer_below_6pct(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=200, seed=42)
        acc     = ev.WERAccumulator("overall")
        for s in samples:
            acc.add(s.reference, s.hypothesis)
        assert acc.micro_wer * 100 <= 6.0, f"Overall WER {acc.micro_wer*100:.2f}% exceeds 6%"

    def test_asr_synthetic_audio_types_present(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=100, seed=42)
        types   = {s.audio_type for s in samples}
        assert "clear"        in types
        assert "mixed_accent" in types
        assert "noisy"        in types

    def test_asr_synthetic_n_samples(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=50, seed=42)
        assert len(samples) == 50

    def test_asr_synthetic_deterministic(self):
        """Same seed → same results."""
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        s1 = ev.generate_synthetic_samples(n=20, seed=7)
        s2 = ev.generate_synthetic_samples(n=20, seed=7)
        assert [s.reference for s in s1] == [s.reference for s in s2]

    def test_nlp_synthetic_deterministic(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        d1 = ev.generate_synthetic_predictions(n=50, target_accuracy=0.942, seed=99)
        d2 = ev.generate_synthetic_predictions(n=50, target_accuracy=0.942, seed=99)
        for dim in ev.DIMS:
            assert d1[dim][0] == d2[dim][0]


# ─────────────────────────────────────────────────────────────────────────────
# TestMetricFunctions
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricFunctions:
    """accuracy_score, macro_f1_score — correctness on known cases."""

    @pytest.fixture(autouse=True)
    def load_nlp(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as m
        self.m = m

    def test_accuracy_perfect(self):
        assert self.m.accuracy_score([0,1,2,3], [0,1,2,3]) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        assert self.m.accuracy_score([0,0,0], [1,1,1]) == pytest.approx(0.0)

    def test_accuracy_half(self):
        assert self.m.accuracy_score([0,1,0,1], [0,0,1,1]) == pytest.approx(0.5)

    def test_accuracy_empty(self):
        assert self.m.accuracy_score([], []) == pytest.approx(0.0)

    def test_f1_perfect(self):
        preds  = [0,1,2,3,4] * 20
        labels = [0,1,2,3,4] * 20
        f1 = self.m.macro_f1_score(preds, labels)
        assert f1 == pytest.approx(1.0)

    def test_f1_zero(self):
        # All predictions wrong, binary case
        preds  = [0] * 10
        labels = [1] * 10
        f1 = self.m.macro_f1_score(preds, labels)
        assert f1 == pytest.approx(0.0)

    def test_f1_skips_absent_classes(self):
        # Only classes 0 and 1 present; classes 2,3,4 absent — should not crash
        preds  = [0,0,1,1]
        labels = [0,1,0,1]
        f1 = self.m.macro_f1_score(preds, labels)
        assert 0.0 <= f1 <= 1.0


class TestSentimentMetrics:
    """per_class_metrics and macro_f1_score for the sentiment evaluator."""

    @pytest.fixture(autouse=True)
    def load(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as m
        self.m = m

    def test_per_class_perfect_classification(self):
        preds  = list(range(4)) * 25
        labels = list(range(4)) * 25
        metrics = self.m.per_class_metrics(preds, labels)
        for m in metrics:
            assert m["precision"] == pytest.approx(100.0)
            assert m["recall"]    == pytest.approx(100.0)
            assert m["f1"]        == pytest.approx(100.0)

    def test_per_class_support_counts(self):
        labels  = [0]*10 + [1]*20 + [2]*5 + [3]*15
        preds   = labels[:]   # perfect
        metrics = self.m.per_class_metrics(preds, labels)
        assert metrics[0]["support"] == 10
        assert metrics[1]["support"] == 20
        assert metrics[2]["support"] == 5
        assert metrics[3]["support"] == 15

    def test_macro_f1_average(self):
        """macro F1 is mean of per-class F1s."""
        preds  = [0,1,2,3] * 100
        labels = [0,1,2,3] * 100
        f1 = self.m.macro_f1_score(preds, labels)
        assert f1 == pytest.approx(100.0)

    def test_accuracy_correct(self):
        preds  = [0,0,1,1,2,2]
        labels = [0,1,1,1,2,3]
        # Correct: pos 0 (0==0), pos 2 (1==1), pos 3 (1==1), pos 4 (2==2) → 4/6 = 0.667
        acc = self.m.accuracy_score(preds, labels)
        assert acc == pytest.approx(4/6, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TestEvaluationCLI
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationCLI:
    """Smoke-test each script's --synthetic mode end-to-end via subprocess."""

    def _run(self, cmd: list[str]) -> tuple[int, str]:
        import subprocess
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(ROOT),
        )
        return result.returncode, result.stdout + result.stderr

    def test_nlp_synthetic_cli_exits_zero(self):
        rc, out = self._run([
            "python3", "nlp_service/evaluate.py",
            "--synthetic", "--n", "200", "--seed", "42",
        ])
        assert rc == 0, f"Non-zero exit:\n{out}"

    def test_nlp_synthetic_cli_shows_pass(self):
        _, out = self._run([
            "python3", "nlp_service/evaluate.py",
            "--synthetic", "--n", "500", "--seed", "42",
        ])
        assert "✓ PASS" in out or "PASS" in out

    def test_nlp_synthetic_cli_saves_json(self, tmp_path):
        out_file = str(tmp_path / "nlp_results.json")
        rc, _ = self._run([
            "python3", "nlp_service/evaluate.py",
            "--synthetic", "--n", "100", "--seed", "1",
            "--output_json", out_file,
        ])
        assert rc == 0
        assert Path(out_file).exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "macro" in data
        assert "per_dimension" in data

    def test_sentiment_synthetic_cli_exits_zero(self):
        rc, out = self._run([
            "python3", "sentiment_service/evaluate_sentiment.py",
            "--synthetic", "--n", "500", "--seed", "42",
        ])
        assert rc == 0, f"Non-zero exit:\n{out}"

    def test_sentiment_synthetic_cli_shows_pass(self):
        _, out = self._run([
            "python3", "sentiment_service/evaluate_sentiment.py",
            "--synthetic", "--n", "1000", "--seed", "42",
        ])
        assert "PASS" in out

    def test_sentiment_synthetic_cli_saves_json(self, tmp_path):
        out_file = str(tmp_path / "sent_results.json")
        rc, _ = self._run([
            "python3", "sentiment_service/evaluate_sentiment.py",
            "--synthetic", "--n", "200", "--seed", "1",
            "--output_json", out_file,
        ])
        assert rc == 0
        with open(out_file) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "per_class" in data
        assert len(data["per_class"]) == 4

    def test_asr_synthetic_cli_exits_zero(self):
        rc, out = self._run([
            "python3", "asr_service/evaluate_asr.py",
            "--synthetic", "--n", "50", "--seed", "42",
        ])
        assert rc == 0, f"Non-zero exit:\n{out}"

    def test_asr_synthetic_cli_shows_pass(self):
        _, out = self._run([
            "python3", "asr_service/evaluate_asr.py",
            "--synthetic", "--n", "100", "--seed", "42",
        ])
        assert "PASS" in out

    def test_asr_synthetic_cli_saves_json(self, tmp_path):
        out_file = str(tmp_path / "asr_results.json")
        rc, _ = self._run([
            "python3", "asr_service/evaluate_asr.py",
            "--synthetic", "--n", "80", "--seed", "1",
            "--output_json", out_file,
        ])
        assert rc == 0
        with open(out_file) as f:
            data = json.load(f)
        assert "overall"  in data
        assert "per_type" in data
        assert "targets_met" in data

    def test_nlp_missing_required_arg_exits_nonzero(self):
        rc, out = self._run(["python3", "nlp_service/evaluate.py"])
        assert rc != 0  # argparse error

    def test_asr_missing_required_arg_exits_nonzero(self):
        rc, _ = self._run(["python3", "asr_service/evaluate_asr.py"])
        assert rc != 0


# ─────────────────────────────────────────────────────────────────────────────
# TestPaperTargets
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperTargets:
    """
    Verify the compute_and_print_results / compute_and_print_wer functions
    set targets_met correctly for pass and fail cases.
    """

    def test_nlp_targets_met_on_passing_data(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        import io, logging
        dim_data = ev.generate_synthetic_predictions(n=1000, target_accuracy=0.950, seed=42)
        with patch("logging.Logger.info"):   # silence output
            results = ev.compute_and_print_results(
                dim_data, "test-pass", verbose=False,
            )
        assert results["targets_met"]["accuracy"] is True
        assert results["targets_met"]["f1"]       is True

    def test_nlp_targets_fail_on_low_accuracy(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        # Force very low accuracy
        dim_data = {d: ([0]*1000, [4]*1000) for d in ev.DIMS}  # all wrong
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_results(dim_data, "test-fail")
        assert results["targets_met"]["accuracy"] is False
        assert results["targets_met"]["f1"]       is False

    def test_sentiment_targets_met_on_passing_data(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        preds, labels = ev.generate_synthetic_predictions(n=2000, target_accuracy=0.918, seed=42)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_results(preds, labels, "test-pass")
        assert results["targets_met"]["accuracy"] is True

    def test_sentiment_targets_fail_on_low_accuracy(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        preds  = [0] * 500
        labels = [3] * 500    # all wrong
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_results(preds, labels, "test-fail")
        assert results["targets_met"]["accuracy"] is False

    def test_asr_targets_met_on_passing_data(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=200, seed=42)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_wer(samples, "test-pass")
        assert results["targets_met"]["wer_clear"]   is True
        assert results["targets_met"]["wer_overall"] is True

    def test_asr_targets_fail_on_high_wer(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        # Manufacture samples with very high WER (all clear, 80% errors)
        samples = ev.generate_synthetic_samples(n=200, seed=42, wer_clear=0.80, wer_mixed=0.80)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_wer(samples, "test-fail")
        assert results["targets_met"]["wer_clear"] is False

    def test_asr_targets_key_structure(self):
        _ensure(ASR_DIR)
        sys.modules.pop("evaluate_asr", None)
        import evaluate_asr as ev
        samples = ev.generate_synthetic_samples(n=100, seed=1)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_wer(samples, "test")
        assert "wer_clear"   in results["targets_met"]
        assert "wer_overall" in results["targets_met"]

    def test_nlp_results_json_structure(self):
        _ensure(NLP_DIR)
        sys.modules.pop("evaluate", None)
        import evaluate as ev
        dim_data = ev.generate_synthetic_predictions(n=100, target_accuracy=0.950, seed=1)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_results(dim_data, "test")
        assert "per_dimension" in results
        assert "macro"         in results
        assert "targets_met"   in results
        for dim in ev.DIMS:
            assert dim in results["per_dimension"]
            assert "accuracy"  in results["per_dimension"][dim]
            assert "macro_f1"  in results["per_dimension"][dim]
            assert "n_samples" in results["per_dimension"][dim]

    def test_sentiment_results_json_structure(self):
        _ensure(SENTIMENT_DIR)
        sys.modules.pop("evaluate_sentiment", None)
        import evaluate_sentiment as ev
        preds, labels = ev.generate_synthetic_predictions(n=200, target_accuracy=0.918, seed=1)
        with patch("logging.Logger.info"):
            results = ev.compute_and_print_results(preds, labels, "test")
        assert "accuracy"    in results
        assert "macro_f1"    in results
        assert "per_class"   in results
        assert "n_samples"   in results
        assert "targets_met" in results
        assert len(results["per_class"]) == 4
        for cls in results["per_class"]:
            assert "class"     in cls
            assert "precision" in cls
            assert "recall"    in cls
            assert "f1"        in cls
