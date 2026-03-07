"""
Evaluation metrics — pure Python, no ML dependencies.

Extracted from evaluate.py so accuracy() and macro_f1()
can be tested without torch/transformers installed.
"""
from collections import defaultdict


def accuracy(preds: list[int], labels: list[int]) -> float:
    """Exact-match accuracy: fraction of predictions equal to the label."""
    if not labels:
        return 0.0
    assert len(preds) == len(labels)
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def macro_f1(preds: list[int], labels: list[int], num_classes: int = 5) -> float:
    """
    Macro-averaged F1 across all score classes (0-4).
    Unweighted mean of per-class F1 — rare classes count equally.
    """
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for pred, label in zip(preds, labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred]  += 1
            fn[label] += 1

    f1_per_class = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall    = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        f1_per_class.append(f1)

    return sum(f1_per_class) / num_classes
