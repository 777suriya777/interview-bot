"""
selector.py — Adaptive question selection engine.

Implements the scoring formula from spec Section 4.3:

    Score(q) = w1*D(q) + w2*W(q) + w3*R(q) + w4*V(q)
    Weights:  w1=0.35, w2=0.30, w3=0.25, w4=0.10  (pilot-tuned)

    D(q) = 1.0 - abs(user_level - q.difficulty) / 4.0   # difficulty proximity
    W(q) = max(0, (avg - topic_score[q.category]) / avg) # weak area boost
    R(q) = 0 if seen this session else 1                 # recency penalty (hard zero)
    V(q) = 1 - (session_category_counts[q.type] / total) # variety reward

EMA difficulty adjustment (alpha=0.3):
    new_perf = 0.3 * latest_overall_score + 0.7 * current_perf_score
    if new_perf > 3.5: difficulty = min(difficulty + 1, 5)
    elif new_perf < 2.0: difficulty = max(difficulty - 1, 1)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Scoring weights (from spec — tuned on 30-user pilot) ──────────────────────
W1_DIFFICULTY  = 0.35   # proximity to user's current level
W2_WEAK_AREA   = 0.30   # boost questions in weak topic areas
W3_RECENCY     = 0.25   # penalise (hard zero) questions seen this session
W4_VARIETY     = 0.10   # reward underrepresented question types

# EMA smoothing factor (spec: alpha = 0.3)
EMA_ALPHA = 0.3

# Difficulty thresholds for adjustment
DIFFICULTY_INCREASE_THRESHOLD = 3.5   # score > 3.5 → increase difficulty
DIFFICULTY_DECREASE_THRESHOLD = 2.0   # score < 2.0 → decrease difficulty
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 5


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class QuestionRecord:
    """
    Lightweight question representation for the scoring engine.
    Populated from DB/Redis cache — no SQLAlchemy dependency.
    """
    id:          str
    text:        str
    type:        str        # 'technical' | 'behavioural' | 'hr'
    difficulty:  int        # 1–5
    category:    str = ""   # e.g. 'data_structures', 'system_design'


@dataclass
class SessionContext:
    """
    All per-session state needed to score questions.
    Stored in Redis under session:{session_id}:state.
    """
    session_id:          str
    user_id:             str
    current_difficulty:  float = 3.0      # starts at 3 (middle of 1–5 scale)
    performance_score:   float = 0.0      # EMA of overall scores
    asked_question_ids:  list[str] = field(default_factory=list)
    category_counts:     dict[str, int] = field(default_factory=dict)
    total_questions:     int = 0


@dataclass
class SelectionResult:
    """Returned by select_next_question()."""
    question_id:      str
    question_text:    str
    question_type:    str
    difficulty:       int
    user_level:       float       # current estimated performance (0–5 scale)
    selection_reason: str         # debug string, e.g. 'weak_area_boost:system_design'
    score:            float = 0.0 # composite score that caused selection


# ── Scoring formula components ─────────────────────────────────────────────────

def score_difficulty_proximity(user_level: float, question_difficulty: int) -> float:
    """
    D(q) = 1.0 - abs(user_level - q.difficulty) / 4.0

    Returns 1.0 when difficulty exactly matches user level, decreasing
    linearly to 0.0 when the gap is 4 levels (max possible).

    Args:
        user_level:          estimated performance level (float 1.0–5.0)
        question_difficulty: integer 1–5

    Returns:
        float in [0.0, 1.0]
    """
    return max(0.0, 1.0 - abs(user_level - question_difficulty) / 4.0)


def score_weak_area(
    question_category: str,
    topic_scores: dict[str, float],
) -> float:
    """
    W(q) = max(0, (avg_score - topic_score[q.category]) / avg_score)

    Boosts questions in topics where the user scores below their average.
    Returns 0.0 if topic has no recorded score (unknown topics → neutral).

    Args:
        question_category: the question's category string
        topic_scores:      dict of topic → EMA score (0–100 scale)

    Returns:
        float in [0.0, 1.0]
    """
    if not topic_scores:
        return 0.0

    avg_score = sum(topic_scores.values()) / len(topic_scores)
    if avg_score <= 0:
        return 0.0

    topic_score = topic_scores.get(question_category)
    if topic_score is None:
        # Unknown topic → neutral (no boost, no penalty)
        return 0.0

    return max(0.0, (avg_score - topic_score) / avg_score)


def score_recency(question_id: str, asked_ids: list[str]) -> float:
    """
    R(q) = 0 if q.id in session_question_ids else 1

    Hard zero for questions already asked this session.
    This is a mandatory exclusion, not just a penalty.

    Args:
        question_id: the question's UUID
        asked_ids:   list of UUIDs asked so far in this session

    Returns:
        0.0 (already asked) or 1.0 (not yet asked)
    """
    return 0.0 if question_id in asked_ids else 1.0


def score_variety(
    question_type: str,
    category_counts: dict[str, int],
    total_questions: int,
) -> float:
    """
    V(q) = 1 - (session_category_counts[q.type] / total_session_questions)

    Rewards question types that are underrepresented in the current session.
    Returns 1.0 when total_questions == 0 (first question — all types equally new).

    Args:
        question_type:    'technical' | 'behavioural' | 'hr'
        category_counts:  dict mapping type → count so far in session
        total_questions:  total questions asked so far

    Returns:
        float in [0.0, 1.0]
    """
    if total_questions <= 0:
        return 1.0
    count = category_counts.get(question_type, 0)
    return max(0.0, 1.0 - (count / total_questions))


def composite_score(
    question: QuestionRecord,
    user_level: float,
    asked_ids: list[str],
    topic_scores: dict[str, float],
    category_counts: dict[str, int],
    total_questions: int,
) -> float:
    """
    Compute the full composite score for a single question.

    Score(q) = w1*D(q) + w2*W(q) + w3*R(q) + w4*V(q)

    Returns 0.0 immediately if the question has been seen (R=0 → hard exclusion
    via the R term, but we explicitly short-circuit for clarity).

    Args:
        question:        QuestionRecord to score
        user_level:      current estimated performance (float 1.0–5.0)
        asked_ids:       list of question IDs already asked this session
        topic_scores:    dict of topic → EMA score (0–100 scale from user_performance)
        category_counts: per-type counts in the current session
        total_questions: total questions asked so far in the session

    Returns:
        composite score float (0.0 = already seen or all weights zero)
    """
    r = score_recency(question.id, asked_ids)
    if r == 0.0:
        return 0.0  # Hard exclusion — already asked this session

    d = score_difficulty_proximity(user_level, question.difficulty)
    w = score_weak_area(question.category, topic_scores)
    v = score_variety(question.type, category_counts, total_questions)

    return W1_DIFFICULTY * d + W2_WEAK_AREA * w + W3_RECENCY * r + W4_VARIETY * v


# ── EMA difficulty tracker ────────────────────────────────────────────────────

def update_performance_ema(
    current_score: float,
    latest_overall: float,
) -> float:
    """
    Update the exponential moving average of performance.

    new_perf = alpha * latest_overall_score + (1 - alpha) * current_perf_score

    The overall score from the NLP service is on a 0–4 scale.
    We normalise it to 0–5 to match the difficulty scale before EMA:
        normalised = latest_overall * (5/4)

    Args:
        current_score:   current EMA score (0–5 scale)
        latest_overall:  latest NLP overall_score (0–4 scale)

    Returns:
        Updated EMA score (0–5 scale, rounded to 3dp)
    """
    # Normalise 0–4 → 0–5 to match difficulty scale
    normalised = latest_overall * (5.0 / 4.0)
    new_score = EMA_ALPHA * normalised + (1.0 - EMA_ALPHA) * current_score
    return round(new_score, 3)


def adjust_difficulty(
    current_difficulty: int,
    performance_score: float,
) -> int:
    """
    Apply threshold rules to adjust difficulty after each answer.

    if new_perf > 3.5: difficulty = min(difficulty + 1, 5)  # increase
    elif new_perf < 2.0: difficulty = max(difficulty - 1, 1) # decrease
    # No change between 2.0 and 3.5

    Note: performance_score is on the 0–5 scale (post-normalisation from EMA).
    The thresholds from spec (3.5 and 2.0) are in the 0–4 NLP score scale,
    but the EMA already normalises them. We rescale thresholds accordingly:
        3.5 on 0-4 scale → 3.5 * (5/4) = 4.375 on 0-5 scale
        2.0 on 0-4 scale → 2.0 * (5/4) = 2.5 on 0-5 scale

    Args:
        current_difficulty: integer 1–5
        performance_score:  EMA score on 0–5 scale

    Returns:
        New difficulty integer 1–5
    """
    high_threshold = DIFFICULTY_INCREASE_THRESHOLD * (5.0 / 4.0)  # 4.375
    low_threshold  = DIFFICULTY_DECREASE_THRESHOLD * (5.0 / 4.0)  # 2.5

    if performance_score > high_threshold:
        return min(current_difficulty + 1, DIFFICULTY_MAX)
    elif performance_score < low_threshold:
        return max(current_difficulty - 1, DIFFICULTY_MIN)
    return current_difficulty


# ── Question selection ────────────────────────────────────────────────────────

def select_next_question(
    questions: list[QuestionRecord],
    context: SessionContext,
    topic_scores: dict[str, float],
    last_scores: Optional[dict[str, int]] = None,
) -> Optional[SelectionResult]:
    """
    Select the best next question from the available pool.

    Algorithm:
      1. If last_scores provided: update EMA, adjust difficulty
      2. Score every question with composite_score()
      3. Return the highest-scoring question
      4. If all questions scored 0 (all seen): return None

    Args:
        questions:    full list of active QuestionRecords (from DB/cache)
        context:      current session context (mutated in-place: EMA + difficulty)
        topic_scores: user's per-topic performance (topic → score 0–100)
        last_scores:  dict from NLP service {content, relevance, completeness, accuracy}

    Returns:
        SelectionResult, or None if no unseen questions available
    """
    # ── Step 1: Update EMA and difficulty ────────────────────────────
    if last_scores:
        # Compute overall from the 4 NLP dimension scores (0–4 each)
        dim_values = [
            last_scores.get("content",      0),
            last_scores.get("relevance",    0),
            last_scores.get("completeness", 0),
            last_scores.get("accuracy",     0),
        ]
        latest_overall = sum(dim_values) / len(dim_values)

        context.performance_score = update_performance_ema(
            context.performance_score,
            latest_overall,
        )
        context.current_difficulty = adjust_difficulty(
            int(round(context.current_difficulty)),
            context.performance_score,
        )

    # ── Step 2: Score all available questions ─────────────────────────
    user_level = context.current_difficulty  # 1–5 scale

    best_question: Optional[QuestionRecord] = None
    best_score    = -1.0
    best_reason   = "random_selection"

    # Normalise topic scores from 0–100 to 0–5 scale for W(q) calculation
    # (The formula uses relative differences, so absolute scale doesn't matter,
    #  but keeping them consistent avoids confusion)
    normalised_topics: dict[str, float] = {
        topic: score / 20.0  # 0–100 → 0–5
        for topic, score in topic_scores.items()
    }

    for q in questions:
        score = composite_score(
            question        = q,
            user_level      = user_level,
            asked_ids       = context.asked_question_ids,
            topic_scores    = normalised_topics,
            category_counts = context.category_counts,
            total_questions = context.total_questions,
        )

        if score > best_score:
            best_score    = score
            best_question = q

    if best_question is None or best_score <= 0.0:
        logger.warning(
            f"No unseen questions available for session {context.session_id}. "
            f"Asked: {len(context.asked_question_ids)}, Pool: {len(questions)}"
        )
        return None

    # ── Step 3: Build selection reason (for debug/logging) ───────────
    weak_topics = [
        topic for topic, score in normalised_topics.items()
        if score < (sum(normalised_topics.values()) / max(len(normalised_topics), 1))
    ]
    if best_question.category in weak_topics:
        reason = f"weak_area_boost:{best_question.category}"
    elif abs(best_question.difficulty - user_level) <= 0.5:
        reason = f"difficulty_match:{best_question.difficulty}"
    else:
        reason = f"composite_score:{best_score:.2f}"

    return SelectionResult(
        question_id      = best_question.id,
        question_text    = best_question.text,
        question_type    = best_question.type,
        difficulty       = best_question.difficulty,
        user_level       = round(user_level, 2),
        selection_reason = reason,
        score            = round(best_score, 4),
    )


def select_fallback_question(
    questions: list[QuestionRecord],
    asked_ids: list[str],
    target_difficulty: int = 3,
) -> Optional[QuestionRecord]:
    """
    Emergency fallback: pick a random unseen question at the target difficulty.
    Used when the adaptive engine times out or the DB is unavailable.

    Tries the exact difficulty first, then relaxes by ±1, ±2, etc.

    Args:
        questions:         available questions (may include seen ones)
        asked_ids:         IDs already asked this session
        target_difficulty: preferred difficulty (default: 3 = medium)

    Returns:
        QuestionRecord or None if no questions available at all
    """
    unseen = [q for q in questions if q.id not in asked_ids]
    if not unseen:
        return None

    # Try to find a question at the target difficulty first
    for delta in range(0, 5):
        candidates = [
            q for q in unseen
            if abs(q.difficulty - target_difficulty) == delta
        ]
        if candidates:
            return random.choice(candidates)

    return random.choice(unseen)
