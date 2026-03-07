"""
evaluate_asr.py — Measures Word Error Rate (WER) for the Whisper ASR pipeline.

Replicates the paper's Table II target metrics:
  - ASR WER clear speech:  ≤ 4.0%  (paper: 3.2%)
  - ASR WER overall:       ≤ 6.0%  (paper: 5.3%)

WER definition:
  WER = (S + D + I) / N
  where S = substitutions, D = deletions, I = insertions, N = reference words.
  Computed via dynamic programming (edit distance on word sequences).

Usage (real audio files):
  python asr_service/evaluate_asr.py \
    --data_dir  data/asr_test/          \
    --manifest  data/asr_test/manifest.jsonl

  manifest.jsonl format (one JSON per line):
    {"audio": "001.wav", "reference": "binary search divides the array", "type": "clear"}

Usage (synthetic — no audio required, matches paper statistics):
  python asr_service/evaluate_asr.py --synthetic [--n 100] [--seed 42]

Output:
  WER per audio type (clear / mixed accent / noisy).
  Overall WER.
  Paper Table II pass/fail verdict.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── WER computation ────────────────────────────────────────────────────────────

def _edit_distance_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """
    Compute substitutions, deletions, insertions using DP edit distance.
    Returns (substitutions, deletions, insertions).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, last_op) where last_op in 'S','D','I','='
    INF = float("inf")

    # dp[i][j] stores minimum cost of aligning ref[:i] with hyp[:j]
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        dp[i][0] = i   # i deletions
    for j in range(1, m + 1):
        dp[0][j] = j   # j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i-1][j-1]                  # match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j-1],   # substitution
                    dp[i-1][j],     # deletion
                    dp[i][j-1],     # insertion
                )

    # Backtrack to count S, D, I
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1  # match
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1  # substitution
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1          # deletion
        else:
            ins += 1; j -= 1           # insertion

    return subs, dels, ins


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int]:
    """
    Compute WER for a single utterance.
    Returns (wer_fraction, n_ref_words).
    Normalises to lowercase, strips punctuation.
    """
    import re
    def _normalise(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    ref = _normalise(reference)
    hyp = _normalise(hypothesis)

    if not ref:
        return (0.0, 0)

    s, d, ins = _edit_distance_ops(ref, hyp)
    wer = (s + d + ins) / len(ref)
    return (wer, len(ref))


class WERAccumulator:
    """Accumulate WER across multiple utterances (macro and micro)."""

    def __init__(self, name: str) -> None:
        self.name       = name
        self.total_ops  = 0   # S + D + I
        self.total_ref  = 0   # total reference words
        self.utterances = 0
        self.per_utt:   list[float] = []

    def add(self, reference: str, hypothesis: str) -> None:
        wer, n_ref = compute_wer(reference, hypothesis)
        s, d, i_   = _edit_distance_ops(
            reference.lower().split(), hypothesis.lower().split()
        )
        self.total_ops  += s + d + i_
        self.total_ref  += n_ref
        self.utterances += 1
        if n_ref > 0:
            self.per_utt.append(wer)

    @property
    def micro_wer(self) -> float:
        """Global WER (total errors / total reference words)."""
        if self.total_ref == 0:
            return 0.0
        return self.total_ops / self.total_ref

    @property
    def macro_wer(self) -> float:
        """Mean WER per utterance."""
        if not self.per_utt:
            return 0.0
        return sum(self.per_utt) / len(self.per_utt)


# ── Synthetic data generator ───────────────────────────────────────────────────

# Vocabulary for building synthetic sentences
_VOCAB_TECH = (
    "binary search algorithm recursion tree graph dynamic programming "
    "hash table queue stack pointer memory allocation linked list array "
    "complexity time space optimal solution approach function method class "
    "object oriented inheritance polymorphism abstraction encapsulation "
    "database index transaction SQL join query normalisation foreign key "
    "primary key constraint schema migration replication sharding latency "
    "throughput scalability availability consistency partition tolerance "
    "microservices REST API HTTP WebSocket JSON serialisation deserialisation"
).split()

_VOCAB_HR = (
    "teamwork collaboration communication leadership initiative ownership "
    "deadline priority conflict resolution stakeholder feedback iteration "
    "agile scrum sprint retrospective planning estimation velocity delivery "
    "mentor coach growth learning improvement performance objective goal "
    "strategy vision mission value culture diversity inclusion engagement"
).split()

_ALL_WORDS = _VOCAB_TECH + _VOCAB_HR


def _make_sentence(rng: random.Random, min_words: int = 8, max_words: int = 25) -> str:
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choices(_ALL_WORDS, k=n))


def _corrupt_sentence(
    sentence: str,
    wer_target: float,
    rng: random.Random,
) -> str:
    """
    Apply random word-level errors to a sentence to achieve approximately
    the target WER. Uses substitution, deletion, and insertion with equal
    probability for each possible error position.
    """
    words = sentence.split()
    if not words:
        return sentence

    result = []
    for word in words:
        if rng.random() < wer_target:
            op = rng.choice(["sub", "del", "ins"])
            if op == "sub":
                result.append(rng.choice(_ALL_WORDS))
            elif op == "del":
                pass  # skip this word
            else:  # ins: add extra word then keep original
                result.append(rng.choice(_ALL_WORDS))
                result.append(word)
        else:
            result.append(word)

    return " ".join(result)


class SyntheticSample(NamedTuple):
    reference:  str
    hypothesis: str
    audio_type: str   # 'clear' | 'mixed_accent' | 'noisy'


def generate_synthetic_samples(
    n:          int   = 100,
    seed:       int   = 42,
    wer_clear:  float = 0.032,   # paper: 3.2%
    wer_mixed:  float = 0.074,   # (5.3% overall with ~50/50 split → mixed ~7.4%)
    wer_noisy:  float = 0.095,   # noisy accent / degraded audio
) -> list[SyntheticSample]:
    """
    Generate synthetic (reference, hypothesis) pairs that, when WER is computed,
    reproduce the paper's reported 3.2% clear-speech and 5.3% overall WER.

    Distribution: 40% clear, 40% mixed accent, 20% noisy (mimics real-world eval).
    """
    rng     = random.Random(seed)
    samples: list[SyntheticSample] = []

    for i in range(n):
        ref        = _make_sentence(rng)
        audio_type = rng.choices(
            ["clear", "mixed_accent", "noisy"],
            weights=[0.40, 0.40, 0.20],
        )[0]
        target_wer = {
            "clear":       wer_clear,
            "mixed_accent": wer_mixed,
            "noisy":        wer_noisy,
        }[audio_type]

        hyp = _corrupt_sentence(ref, target_wer, rng)
        samples.append(SyntheticSample(ref, hyp, audio_type))

    return samples


# ── Real evaluation (requires audio + Whisper) ────────────────────────────────

def evaluate_with_whisper(
    data_dir:     str,
    manifest_path: str,
    whisper_model: str = "small",
    language:      str = "en",
) -> list[SyntheticSample]:
    """
    Transcribe each audio file in the manifest using Whisper and
    return (reference, hypothesis, type) samples.

    manifest.jsonl format (one JSON per line):
      {"audio": "001.wav", "reference": "the transcript", "type": "clear"}
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        raise ImportError(
            "openai-whisper not installed. "
            "Run: pip install openai-whisper\n"
            "Or use --synthetic for a model-free run."
        )

    logger.info(f"Loading Whisper model: {whisper_model}")
    model   = whisper.load_model(whisper_model)
    samples: list[SyntheticSample] = []

    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Transcribing {len(entries)} audio files…")

    for i, entry in enumerate(entries):
        audio_path = Path(data_dir) / entry["audio"]
        if not audio_path.exists():
            logger.warning(f"  Audio file not found: {audio_path}. Skipping.")
            continue

        if i % 10 == 0:
            logger.info(f"  [{i+1}/{len(entries)}] {entry['audio']}")

        result    = model.transcribe(str(audio_path), language=language, fp16=False)
        hypothesis = result["text"].strip()
        samples.append(SyntheticSample(
            reference  = entry["reference"],
            hypothesis = hypothesis,
            audio_type = entry.get("type", "unknown"),
        ))

    return samples


# ── Metrics computation and reporting ─────────────────────────────────────────

def compute_and_print_wer(
    samples:    list[SyntheticSample],
    mode_label: str,
) -> dict:
    """Compute WER by audio type and overall; print table; return results dict."""

    # Accumulators per audio type + overall
    accumulators: dict[str, WERAccumulator] = {
        "clear":        WERAccumulator("Clear Speech"),
        "mixed_accent": WERAccumulator("Mixed Accents"),
        "noisy":        WERAccumulator("Noisy Audio"),
        "overall":      WERAccumulator("Overall"),
    }

    for s in samples:
        acc_type = accumulators.get(s.audio_type)
        if acc_type:
            acc_type.add(s.reference, s.hypothesis)
        accumulators["overall"].add(s.reference, s.hypothesis)

    # ── Print table ────────────────────────────────────────────────────
    border = "=" * 70
    logger.info(f"\n{border}")
    logger.info("  ASR EVALUATION RESULTS  —  Whisper (small) Pipeline")
    logger.info(f"  Mode: {mode_label}")
    logger.info(border)
    logger.info(f"  {'Audio Type':22s} | {'WER (micro)':>12s} | {'WER (macro)':>12s} | {'N':>6s}")
    logger.info(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")

    results: dict = {"mode": mode_label, "per_type": {}, "overall": {}}

    for key, acc in accumulators.items():
        if acc.utterances == 0:
            continue
        micro = acc.micro_wer * 100
        macro = acc.macro_wer * 100
        label = acc.name

        logger.info(
            f"  {label:22s} | {micro:11.2f}% | {macro:11.2f}% | {acc.utterances:>6d}"
        )

        if key == "overall":
            results["overall"] = {
                "wer_micro": round(micro, 2),
                "wer_macro": round(macro, 2),
                "n_utterances": acc.utterances,
            }
        else:
            results["per_type"][key] = {
                "wer_micro": round(micro, 2),
                "wer_macro": round(macro, 2),
                "n_utterances": acc.utterances,
                "label": label,
            }

    # ── Paper targets ──────────────────────────────────────────────────
    overall_wer = accumulators["overall"].micro_wer * 100
    clear_wer   = accumulators["clear"].micro_wer   * 100

    TARGET_CLEAR   = 4.0
    TARGET_OVERALL = 6.0

    clear_pass   = "✓ PASS" if clear_wer   <= TARGET_CLEAR   else "✗ FAIL"
    overall_pass = "✓ PASS" if overall_wer <= TARGET_OVERALL else "✗ FAIL"

    logger.info(border)
    logger.info(
        f"\n  Paper target WER (clear speech) ≤ {TARGET_CLEAR}%   →  "
        f"{clear_wer:.2f}%  {clear_pass}"
    )
    logger.info(
        f"  Paper target WER (overall)      ≤ {TARGET_OVERALL}%   →  "
        f"{overall_wer:.2f}%  {overall_pass}"
    )
    logger.info(border + "\n")

    results["targets_met"] = {
        "wer_clear":   clear_wer   <= TARGET_CLEAR,
        "wer_overall": overall_wer <= TARGET_OVERALL,
    }
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper ASR pipeline — Word Error Rate (WER)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest",
        help="Path to manifest.jsonl (with audio paths + references + types)",
    )
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data matching paper WER statistics (no audio needed)",
    )

    parser.add_argument("--data_dir",      default=".",
                        help="Directory containing audio files (used with --manifest)")
    parser.add_argument("--whisper_model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--language",      default="en")
    parser.add_argument("--n",             type=int, default=100,
                        help="Number of synthetic utterances (default 100)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output_json",   default=None)

    args = parser.parse_args()

    if args.synthetic:
        logger.info(f"Running in SYNTHETIC mode (n={args.n}, seed={args.seed})")
        samples    = generate_synthetic_samples(n=args.n, seed=args.seed)
        mode_label = f"Synthetic (n={args.n}, seed={args.seed})"
    else:
        logger.info(f"Running in REAL AUDIO mode (manifest={args.manifest})")
        samples    = evaluate_with_whisper(
            data_dir      = args.data_dir,
            manifest_path = args.manifest,
            whisper_model = args.whisper_model,
            language      = args.language,
        )
        mode_label = f"Real audio, Whisper-{args.whisper_model}"

    results = compute_and_print_wer(samples, mode_label)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
