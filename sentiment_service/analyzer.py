"""
analyzer.py — Lexical analysis for confidence scoring.

Torch-free module: detects hedging phrases, counts filler words,
and generates actionable feedback text. Extracted as a separate
module so it can be tested and imported without PyTorch.

These features correspond to Table: Feature Types in the spec:
  - Lexical: hedging words, filler words
  - Syntactic: sentence fragments, trailing question marks
  - Semantic: vague word choices (handled by BiLSTM, not rule-based)
"""
from __future__ import annotations

import re

# ── Hedging Phrases ───────────────────────────────────────────────────────────
# From spec Table: Lexical features that annotators used to label Low/Anxious.
# Ordered longest-first so multi-word phrases match before substrings.
HEDGING_PHRASES: list[str] = [
    # Multi-word (check these first to avoid partial matches)
    "i'm not sure",
    "i am not sure",
    "i'm not entirely sure",
    "i'm not totally sure",
    "i don't really know",
    "i do not really know",
    "i'm not 100%",
    "i'm not a hundred percent",
    "not entirely sure",
    "not totally sure",
    "to be honest",
    "if i'm not mistaken",
    "if i recall correctly",
    "something like that",
    "kind of",
    "sort of",
    "i guess",
    "i think",
    "i believe",
    "i suppose",
    "i feel like",
    "i'm pretty sure",
    "i am pretty sure",
    "probably",
    "possibly",
    "perhaps",
    "maybe",
    # Single-word fillers
    "basically",
    "literally",
    "actually",
    "obviously",
    "clearly",
]

# ── Filler Words ──────────────────────────────────────────────────────────────
# Spec explicitly lists: 'um', 'uh', 'like', 'you know'
FILLER_WORDS: list[str] = [
    "um",
    "uh",
    "uhh",
    "umm",
    "er",
    "err",
    "like",
    "you know",
    "right",
    "okay so",
    "so basically",
]


# ── Detection Functions ───────────────────────────────────────────────────────

def detect_hedging(transcript: str) -> list[dict[str, int | str]]:
    """
    Find all hedging phrase occurrences in the transcript.

    Returns a list of dicts: [{"text": "I think", "start_char": 12}, ...]
    Matching is case-insensitive. Each distinct occurrence is reported
    (same phrase can appear multiple times).
    """
    lower = transcript.lower()
    found: list[dict] = []

    for phrase in HEDGING_PHRASES:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx == -1:
                break
            # Preserve original casing from the transcript
            found.append({"text": transcript[idx: idx + len(phrase)], "start_char": idx})
            start = idx + 1  # allow overlapping matches

    # Sort by position so client receives them in document order
    found.sort(key=lambda x: x["start_char"])
    return found


def count_filler_words(transcript: str) -> int:
    """
    Count total filler word occurrences in the transcript.
    Uses word-boundary matching so 'like' doesn't match 'likewise'.
    """
    lower = transcript.lower()
    total = 0
    for filler in FILLER_WORDS:
        if " " in filler:
            # Multi-word filler: simple substring count
            total += lower.count(filler)
        else:
            # Single word: use word boundary regex to avoid false positives
            total += len(re.findall(r"\b" + re.escape(filler) + r"\b", lower))
    return total


# ── Feedback Text Generation ──────────────────────────────────────────────────

# Spec labels: high=0, moderate=1, low=2, anxious=3
LABEL_NAMES = {0: "high", 1: "moderate", 2: "low", 3: "anxious"}
LABEL_IDS   = {"high": 0, "moderate": 1, "low": 2, "anxious": 3}


def build_confidence_feedback(
    label: str,
    hedging_count: int,
    filler_count: int,
) -> str:
    """
    Generate a 1–2 sentence actionable tip based on the confidence label
    and detected linguistic markers.

    Args:
        label:         'high' | 'moderate' | 'low' | 'anxious'
        hedging_count: number of hedging phrases detected
        filler_count:  number of filler words detected

    Returns:
        Feedback string suitable for display to the candidate.
    """
    tips: dict[str, str] = {
        "high": (
            "Great delivery — you spoke with confidence and conviction. "
            "Keep using concrete, assertive language."
        ),
        "moderate": (
            "Good delivery overall. "
            + (
                f"You used {hedging_count} hedging phrase(s) — try replacing "
                "'I think' or 'I believe' with direct assertions to sound more authoritative."
                if hedging_count > 1
                else "Minor hedging detected — aim for more assertive phrasing."
            )
        ),
        "low": (
            "Your delivery lacked confidence. "
            + (
                f"You used {hedging_count} hedging phrase(s) — phrases like 'I think', "
                "'I guess', and 'kind of' undermine your credibility. "
                "State facts directly: instead of 'I think it uses hashing', say 'It uses hashing'."
                if hedging_count > 2
                else "Replace uncertain language with direct statements. "
                     "Practise stating your answers assertively, even when uncertain."
            )
        ),
        "anxious": (
            "Your response showed signs of anxiety. "
            + (
                f"You used {filler_count} filler word(s) and {hedging_count} hedging phrase(s). "
                if filler_count > 0
                else f"You used {hedging_count} hedging phrase(s). "
            )
            + "Slow down, pause before answering, and use the STAR method to "
              "structure your response — this will help you sound more composed."
        ),
    }
    return tips.get(label, "Keep practising your delivery for a more confident performance.")


def build_improvement_tips(
    nlp_scores: dict[str, int],
    confidence_label: str,
    hedging_count: int,
    filler_count: int,
    delivery_flags: list[str],
) -> list[str]:
    """
    Build the list of improvement_tips sent in the FEEDBACK WebSocket message.
    Combines NLP content tips with delivery/confidence tips.
    Returns 2-4 actionable strings.

    Args:
        nlp_scores:        dict with content/relevance/completeness/accuracy scores
        confidence_label:  'high'|'moderate'|'low'|'anxious'
        hedging_count:     hedging phrases detected
        filler_count:      filler words detected
        delivery_flags:    list of ASR flag strings
    """
    tips: list[str] = []

    # ── Content tips (from NLP scores) ──────────────────────────────
    content_tips = {
        "content": {
            0: "Add technical substance — define key concepts and give concrete examples.",
            1: "Expand your technical detail; mention algorithms, data structures, or trade-offs.",
            2: "Good content — consider adding an edge case or real-world application.",
        },
        "relevance": {
            0: "Re-read the question and focus directly on what is being asked.",
            1: "Stay on-topic — your answer drifted from the core question.",
            2: "Mostly relevant — cut any tangential information.",
        },
        "completeness": {
            0: "Your answer is very incomplete — address all parts of the question.",
            1: "Cover more aspects of the question — several key points are missing.",
            2: "Nearly complete — add 1–2 more supporting details.",
        },
        "accuracy": {
            0: "Review the core concepts — your answer contained factual errors.",
            1: "Double-check your facts and terminology before answering.",
            2: "Mostly accurate — verify the specific claim you're uncertain about.",
        },
    }

    # Add tip for the weakest NLP dimension (if score ≤ 2)
    weakest_dim = min(nlp_scores, key=lambda k: nlp_scores[k])
    if nlp_scores[weakest_dim] <= 2:
        dim_tip = content_tips.get(weakest_dim, {}).get(nlp_scores[weakest_dim])
        if dim_tip:
            tips.append(dim_tip)

    # ── Confidence tip ───────────────────────────────────────────────
    if confidence_label in ("low", "anxious"):
        tips.append(
            build_confidence_feedback(confidence_label, hedging_count, filler_count)
        )
    elif confidence_label == "moderate" and hedging_count > 2:
        tips.append(
            f"Reduce hedging: you used {hedging_count} uncertain phrases. "
            "Replace 'I think X' with 'X' for a more confident tone."
        )

    # ── Delivery tips (from ASR flags) ──────────────────────────────
    delivery_tip_map = {
        "fast_speech":       "Slow down your speaking rate — aim for 130–150 WPM for clarity.",
        "nervous_pitch":     "Your pitch varied significantly — try to speak in a steady, measured tone.",
        "excessive_pauses":  "Reduce excessive pausing — practise transitions between ideas.",
        "low_volume":        "Vary your emphasis to sound more engaged and confident.",
        "too_short":         "Expand your answer — aim for at least 60–90 words to fully address the question.",
    }
    for flag in delivery_flags:
        tip = delivery_tip_map.get(flag)
        if tip and tip not in tips:
            tips.append(tip)

    # Return up to 3 most relevant tips (avoid overwhelming the candidate)
    return tips[:3] if tips else ["Good effort — keep practising for even better results."]


# Re-export label maps here so tests that can't import model.py (no torch)
# can still get the label definitions.
IDX_TO_LABEL = {0: "high", 1: "moderate", 2: "low", 3: "anxious"}
LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}
