/**
 * handlers/answer.js — Answer processing pipeline.
 *
 * Implements the spec Section 5.1 flow exactly:
 *
 *   1. ASR (voice only)           → transcript + delivery features
 *   2. NLP + Sentiment in parallel (Promise.all) → scores + confidence
 *   3. Persist answer to DB       (with retry x3)
 *   4. Adaptive engine            → next question
 *   5. Send FEEDBACK to client
 *
 * Timeout behaviour (spec Section 5.4):
 *   NLP timeout    → neutral scores (2,2,2,2) + warning in feedback
 *   ASR timeout    → ERROR frame, do not proceed
 *   Adaptive timeout→ random medium-difficulty fallback from question bank
 *
 * NLP + Sentiment calls are always parallel (Promise.all) — never sequential.
 * ASR must complete before NLP/Sentiment because they need the transcript.
 */

const crypto  = require('crypto');
const db      = require('../db/queries');
const redis   = require('../cache/redis');
const {
  callNLP,
  callASR,
  callSentiment,
  callAdaptive,
} = require('../services');
const { saveSessionState } = require('./session');

// ── Neutral fallback scores (spec: return when NLP times out) ─────────────────
const NEUTRAL_SCORES = {
  content_score      : 2,
  relevance_score    : 2,
  completeness_score : 2,
  accuracy_score     : 2,
  overall_score      : 2.0,
  feedback_summary   : 'Evaluation temporarily unavailable.',
  _was_fallback      : true,
};

// ── NLP cache (Redis) — SHA-256 hash of question+answer, TTL 1h ───────────────

/**
 * Build the Redis cache key for an NLP result.
 * Key = SHA-256(question + '\x00' + answer) — prevents collision between
 * question and answer strings by using a null-byte separator.
 */
function nlpCacheKey(question, answer) {
  const hash = crypto
    .createHash('sha256')
    .update(`${question}\x00${answer}`)
    .digest('hex');
  return `nlp:cache:${hash}`;
}

async function getNlpCache(question, answer) {
  const key = nlpCacheKey(question, answer);
  const cached = await redis.get(key).catch(() => null);
  if (cached) {
    try { return JSON.parse(cached); } catch { /* corrupt cache */ }
  }
  return null;
}

async function setNlpCache(question, answer, result) {
  const key = nlpCacheKey(question, answer);
  await redis.set(key, JSON.stringify(result), 'EX', 3_600).catch(() => {});
}

// ── Fallback question selection (when adaptive engine times out) ──────────────

/**
 * Select a random unseen medium-difficulty question from the question bank.
 * Uses the Redis question_bank:active cache if available.
 *
 * @param {string[]} askedIds IDs already asked this session
 * @returns {object|null} question row, or null if none available
 */
async function fallbackQuestion(askedIds) {
  let questions = [];

  // Try Redis cache first
  const cached = await redis.get('question_bank:active').catch(() => null);
  if (cached) {
    try {
      questions = JSON.parse(cached);
    } catch { /* fall through to DB */ }
  }

  // Fall back to DB
  if (!questions.length) {
    questions = await db.getActiveQuestions().catch(() => []);
  }

  if (!questions.length) return null;

  const askedSet = new Set(askedIds);
  const unseen   = questions.filter((q) => !askedSet.has(q.id?.toString()));
  if (!unseen.length) return null;

  // Prefer difficulty 3 (medium), then relax to ±1, ±2
  for (let delta = 0; delta <= 4; delta++) {
    const candidates = unseen.filter((q) => Math.abs(q.difficulty - 3) === delta);
    if (candidates.length) {
      return candidates[Math.floor(Math.random() * candidates.length)];
    }
  }
  return unseen[0];
}

// ── Improvement tip builder ───────────────────────────────────────────────────

/**
 * Build a list of actionable improvement tips from the evaluation results.
 * Returns 1–3 tips as plain strings (shown to user in FEEDBACK frame).
 *
 * @param {object} nlpResult
 * @param {object} sentResult
 * @param {object|null} asrResult
 * @returns {string[]}
 */
function buildTips(nlpResult, sentResult, asrResult) {
  const tips = [];

  // NLP-based tips
  if (nlpResult.content_score <= 1) {
    tips.push('Focus on directly addressing the core concepts in the question.');
  } else if (nlpResult.completeness_score <= 1) {
    tips.push('Expand your answer — aim for 3–4 key points with concrete examples.');
  }
  if (nlpResult.relevance_score <= 1) {
    tips.push('Stay focused on what the question is specifically asking.');
  }

  // Sentiment-based tip
  if (sentResult.confidence_label === 'low' || sentResult.confidence_label === 'anxious') {
    tips.push(sentResult.feedback_text || 'Speak more confidently — avoid hedging phrases like "I think" or "maybe".');
  }

  // ASR delivery tips (voice only)
  if (asrResult && asrResult.delivery_flags) {
    for (const flag of asrResult.delivery_flags) {
      if (flag.includes('fast')) {
        tips.push('You spoke quickly — try slowing to 130–150 WPM for clarity.');
        break;
      }
      if (flag.includes('pitch')) {
        tips.push('Significant pitch variation detected — try speaking more evenly.');
        break;
      }
    }
  }

  // Always include NLP feedback if available and no other tips
  if (!tips.length && nlpResult.feedback_summary && !nlpResult._was_fallback) {
    tips.push(nlpResult.feedback_summary);
  }

  return tips.slice(0, 3); // spec: max ~3 tips per answer
}

// ── Main answer handler ───────────────────────────────────────────────────────

/**
 * Process a SUBMIT_TEXT or SUBMIT_VOICE message.
 *
 * @param {WebSocket} ws
 * @param {object}    session   from ws.sessionState
 * @param {object}    payload   from the WS message
 * @param {string}    inputMode 'text' | 'voice'
 */
async function handleAnswerSubmit(ws, session, payload, inputMode) {
  const { question_id: questionId, answer_text: answerText, audio_b64: audioB64 } = payload;

  if (!questionId) {
    return sendError(ws, 'invalid_payload', 'question_id is required');
  }

  // ── Step 1: ASR (voice only) ────────────────────────────────────────────────
  let transcript = answerText || '';
  let asrResult  = null;

  if (inputMode === 'voice') {
    if (!audioB64) {
      return sendError(ws, 'invalid_payload', 'audio_b64 is required for voice answers');
    }

    asrResult = await callASR(audioB64);

    // ASR timeout or hard error → abort, prompt re-record
    if (!asrResult || asrResult._error) {
      const code    = asrResult?._error || 'asr_error';
      const message = code === 'asr_timeout'
        ? 'Recording timed out — please re-record or type your answer instead.'
        : (asrResult?._message || 'Audio processing failed. Please try again.');
      return sendError(ws, code, message);
    }

    transcript = asrResult.transcript || '';
  }

  // ── Step 2: Load question text (needed for NLP evaluation) ─────────────────
  const question = await db.getQuestion(questionId).catch(() => null);
  if (!question) {
    return sendError(ws, 'question_not_found', `Question ${questionId} not found`);
  }

  // ── Step 3: NLP + Sentiment in parallel (Promise.all) ──────────────────────
  // Check NLP cache first — if same question+answer was evaluated before,
  // reuse result to avoid redundant GPU calls (e.g. after network retry)
  let nlpResult = await getNlpCache(question.text, transcript);
  let sentResult;

  if (nlpResult) {
    // Cache hit — still need to run sentiment (it's fast and cache isn't worth it)
    [, sentResult] = await Promise.all([
      Promise.resolve(nlpResult),
      callSentiment(transcript),
    ]);
  } else {
    // Cache miss — run both in parallel
    [nlpResult, sentResult] = await Promise.all([
      callNLP(question.text, transcript),
      callSentiment(transcript),
    ]);

    if (!nlpResult) {
      // NLP timed out — use neutral fallback scores (spec Section 5.4)
      nlpResult = { ...NEUTRAL_SCORES };
    } else {
      // Cache the successful NLP result for 1h
      await setNlpCache(question.text, transcript, nlpResult);
    }
  }

  // Sentinel result from sentiment on error is already a safe fallback object
  sentResult = sentResult || {
    confidence_label  : 'moderate',
    hedging_phrases   : [],
    filler_word_count : 0,
    feedback_text     : '',
  };

  // ── Step 4: Persist answer to DB (retry x3 with exponential backoff) ────────
  const answerRecord = {
    sessionId          : session.sessionId,
    questionId,
    inputMode,
    transcript,
    contentScore       : nlpResult.content_score       ?? 2,
    relevanceScore     : nlpResult.relevance_score      ?? 2,
    completenessScore  : nlpResult.completeness_score   ?? 2,
    accuracyScore      : nlpResult.accuracy_score       ?? 2,
    overallScore       : nlpResult.overall_score        ?? 2.0,
    confidenceLabel    : sentResult.confidence_label    || 'moderate',
    hedgingFlags       : sentResult.hedging_phrases     || [],
    deliveryFlags      : asrResult?.delivery_flags      || [],
    speakingRateWpm    : asrResult?.speaking_rate_wpm   ?? null,
    pitchStd           : asrResult?.pitch_std           ?? null,
  };

  await db.withRetry(() => db.insertAnswer(answerRecord)).catch((err) => {
    // Log but don't fail the request — user should still get feedback
    console.error(`[Answer] DB insert failed after retries: ${err.message}`);
  });

  // ── Step 5: Adaptive engine selects next question ───────────────────────────
  const lastScores = {
    content      : nlpResult.content_score       ?? 2,
    relevance    : nlpResult.relevance_score      ?? 2,
    completeness : nlpResult.completeness_score   ?? 2,
    accuracy     : nlpResult.accuracy_score       ?? 2,
  };

  // Track this question as asked and update session state
  if (!session.askedIds.includes(questionId)) {
    session.askedIds.push(questionId);
  }

  let nextQuestion = await callAdaptive({
    userId             : session.userId,
    sessionId          : session.sessionId,
    lastScores,
    sessionQuestionIds : session.askedIds,
  });

  if (!nextQuestion) {
    // Adaptive engine timed out — use local fallback (spec Section 5.4)
    console.warn(`[Adaptive] Timeout — using fallback question for session ${session.sessionId}`);
    const fallbackQ = await fallbackQuestion(session.askedIds);
    if (fallbackQ) {
      nextQuestion = {
        question_id   : fallbackQ.id,
        question_text : fallbackQ.text,
        question_type : fallbackQ.type,
        difficulty    : fallbackQ.difficulty,
        user_level    : session.difficulty || 3,
        selection_reason: 'fallback_random',
      };
    }
  }

  // Persist updated session state to Redis
  await saveSessionState(session).catch(() => {});

  // ── Step 6: Send FEEDBACK to client ─────────────────────────────────────────
  const feedbackPayload = {
    scores: {
      content      : nlpResult.content_score       ?? 2,
      relevance    : nlpResult.relevance_score      ?? 2,
      completeness : nlpResult.completeness_score   ?? 2,
      accuracy     : nlpResult.accuracy_score       ?? 2,
      overall      : nlpResult.overall_score        ?? 2.0,
    },
    confidence_label : sentResult.confidence_label || 'moderate',
    hedging_words    : sentResult.hedging_phrases  || [],
    delivery_flags   : asrResult?.delivery_flags   || [],  // [] for text input
    improvement_tips : buildTips(nlpResult, sentResult, asrResult),
    next_question    : nextQuestion
      ? {
          id         : nextQuestion.question_id,
          text       : nextQuestion.question_text,
          type       : nextQuestion.question_type,
          difficulty : nextQuestion.difficulty,
        }
      : null,
  };

  // Add warning if NLP used fallback scores
  if (nlpResult._was_fallback) {
    feedbackPayload._warning = 'Evaluation partially unavailable — showing estimated scores.';
  }

  ws.send(JSON.stringify({ type: 'FEEDBACK', payload: feedbackPayload }));
}

// ── FLAG_QUESTION handler ─────────────────────────────────────────────────────

/**
 * Handle FLAG_QUESTION message — increments question's flag_count in DB.
 *
 * @param {WebSocket} ws
 * @param {object}    payload  { question_id, reason }
 */
async function handleFlagQuestion(ws, payload) {
  const { question_id: questionId } = payload;
  if (!questionId) return;

  await db.flagQuestion(questionId).catch((err) => {
    console.warn(`[Flag] Failed to flag question ${questionId}: ${err.message}`);
  });

  // No response needed — fire and forget
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function sendError(ws, code, message) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify({ type: 'ERROR', payload: { code, message } }));
  }
}

module.exports = {
  handleAnswerSubmit,
  handleFlagQuestion,
  buildTips,
  fallbackQuestion,
  nlpCacheKey,
  NEUTRAL_SCORES,
};
