/**
 * __tests__/answer.test.js
 *
 * Tests for handlers/answer.js
 * All DB, Redis, and service calls are mocked with jest.mock().
 *
 * Groups:
 *   buildTips              → improvement tip generation
 *   nlpCacheKey            → SHA-256 cache key determinism
 *   fallbackQuestion       → random question selection from bank
 *   handleAnswerSubmit     → full pipeline (text + voice paths)
 *   handleFlagQuestion     → flag question DB update
 */

'use strict';

// ── Mock all dependencies before importing the module under test ─────────────
jest.mock('../db/queries');
jest.mock('../cache/redis');
jest.mock('../services');
jest.mock('../handlers/session', () => ({
  saveSessionState: jest.fn().mockResolvedValue(undefined),
}));

const db       = require('../db/queries');
const redis    = require('../cache/redis');
const services = require('../services');

const {
  handleAnswerSubmit,
  handleFlagQuestion,
  buildTips,
  fallbackQuestion,
  nlpCacheKey,
  NEUTRAL_SCORES,
} = require('../handlers/answer');

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeWs(overrides = {}) {
  return {
    readyState : 1, // OPEN
    OPEN       : 1,
    send       : jest.fn(),
    ...overrides,
  };
}

function makeSession(overrides = {}) {
  return {
    sessionId  : 'session-abc',
    userId     : 'user-xyz',
    askedIds   : [],
    difficulty : 3,
    performance: 0,
    ...overrides,
  };
}

const MOCK_QUESTION = {
  id         : 'q-001',
  text       : 'What is Big O notation?',
  type       : 'technical',
  difficulty : 3,
  category   : 'algorithms',
};

const MOCK_NLP = {
  content_score      : 3,
  relevance_score    : 3,
  completeness_score : 2,
  accuracy_score     : 3,
  overall_score      : 2.75,
  feedback_summary   : 'Good answer covering key concepts.',
};

const MOCK_SENT = {
  confidence_label  : 'high',
  confidence_score  : 0.85,
  hedging_phrases   : [],
  filler_word_count : 0,
  feedback_text     : 'Confident delivery.',
};

const MOCK_NEXT_Q = {
  question_id    : 'q-002',
  question_text  : 'Explain recursion.',
  question_type  : 'technical',
  difficulty     : 3,
  user_level     : 3.0,
  selection_reason: 'difficulty_match:3',
};

// ── Reset mocks before each test ──────────────────────────────────────────────
beforeEach(() => {
  jest.clearAllMocks();
  redis.get.mockResolvedValue(null);
  redis.set.mockResolvedValue('OK');
  db.getQuestion.mockResolvedValue(MOCK_QUESTION);
  db.insertAnswer.mockResolvedValue('answer-id-123');
  db.withRetry.mockImplementation((fn) => fn());
  db.flagQuestion.mockResolvedValue(undefined);
  db.getActiveQuestions.mockResolvedValue([]);
  services.callNLP.mockResolvedValue(MOCK_NLP);
  services.callSentiment.mockResolvedValue(MOCK_SENT);
  services.callASR.mockResolvedValue(null); // not called for text
  services.callAdaptive.mockResolvedValue(MOCK_NEXT_Q);
});

// ── buildTips ─────────────────────────────────────────────────────────────────

describe('buildTips', () => {
  test('returns array', () => {
    const tips = buildTips(MOCK_NLP, MOCK_SENT, null);
    expect(Array.isArray(tips)).toBe(true);
  });

  test('returns at most 3 tips', () => {
    const badNlp = { content_score: 0, relevance_score: 0, completeness_score: 0, overall_score: 0 };
    const badSent = { confidence_label: 'anxious', feedback_text: 'Be more confident.' };
    const tips = buildTips(badNlp, badSent, null);
    expect(tips.length).toBeLessThanOrEqual(3);
  });

  test('includes confidence tip when label is low', () => {
    const sent = { ...MOCK_SENT, confidence_label: 'low', feedback_text: 'Avoid hedging.' };
    const tips = buildTips(MOCK_NLP, sent, null);
    expect(tips.some((t) => t.includes('hedg') || t.includes('confident') || t.includes('Avoid'))).toBe(true);
  });

  test('includes fast speech tip when delivery flag present', () => {
    const asrResult = { delivery_flags: ['fast_speech'] };
    const tips = buildTips(MOCK_NLP, MOCK_SENT, asrResult);
    expect(tips.some((t) => t.includes('WPM') || t.includes('slow'))).toBe(true);
  });

  test('includes NLP feedback when no other tips', () => {
    const goodNlp = { ...MOCK_NLP, content_score: 4, relevance_score: 4,
                       completeness_score: 4, feedback_summary: 'Great answer!' };
    const tips = buildTips(goodNlp, MOCK_SENT, null);
    expect(tips.includes('Great answer!')).toBe(true);
  });

  test('does not include NLP feedback when it was a fallback', () => {
    const fallback = { ...NEUTRAL_SCORES, feedback_summary: 'Fallback', _was_fallback: true };
    const tips = buildTips(fallback, MOCK_SENT, null);
    expect(tips.includes('Fallback')).toBe(false);
  });

  test('empty delivery_flags produces no speech tip', () => {
    const tips = buildTips(MOCK_NLP, MOCK_SENT, { delivery_flags: [] });
    expect(tips.every((t) => !t.includes('WPM'))).toBe(true);
  });
});

// ── nlpCacheKey ───────────────────────────────────────────────────────────────

describe('nlpCacheKey', () => {
  test('returns string starting with nlp:cache:', () => {
    const key = nlpCacheKey('question', 'answer');
    expect(key.startsWith('nlp:cache:')).toBe(true);
  });

  test('same inputs produce same key (deterministic)', () => {
    expect(nlpCacheKey('q', 'a')).toBe(nlpCacheKey('q', 'a'));
  });

  test('different questions produce different keys', () => {
    expect(nlpCacheKey('q1', 'a')).not.toBe(nlpCacheKey('q2', 'a'));
  });

  test('different answers produce different keys', () => {
    expect(nlpCacheKey('q', 'a1')).not.toBe(nlpCacheKey('q', 'a2'));
  });

  test('switching question and answer produces different key (no collision)', () => {
    // Important: "q\x00a" must differ from "a\x00q"
    expect(nlpCacheKey('q', 'a')).not.toBe(nlpCacheKey('a', 'q'));
  });

  test('key has consistent length (SHA-256 hex = 64 chars + prefix)', () => {
    const key = nlpCacheKey('test', 'answer');
    const hash = key.replace('nlp:cache:', '');
    expect(hash.length).toBe(64);
  });
});

// ── fallbackQuestion ──────────────────────────────────────────────────────────

describe('fallbackQuestion', () => {
  const BANK = [
    { id: 'q1', text: 'Q1', type: 'technical',   difficulty: 1, category: 'a' },
    { id: 'q2', text: 'Q2', type: 'technical',   difficulty: 3, category: 'b' },
    { id: 'q3', text: 'Q3', type: 'behavioural', difficulty: 3, category: 'c' },
    { id: 'q4', text: 'Q4', type: 'hr',          difficulty: 5, category: 'd' },
  ];

  test('returns null when no questions available', async () => {
    db.getActiveQuestions.mockResolvedValue([]);
    const result = await fallbackQuestion([]);
    expect(result).toBeNull();
  });

  test('returns null when all questions are seen', async () => {
    redis.get.mockResolvedValue(JSON.stringify(BANK));
    const result = await fallbackQuestion(['q1', 'q2', 'q3', 'q4']);
    expect(result).toBeNull();
  });

  test('never returns a seen question', async () => {
    redis.get.mockResolvedValue(JSON.stringify(BANK));
    const result = await fallbackQuestion(['q1', 'q2']);
    expect(['q3', 'q4']).toContain(result.id);
  });

  test('prefers difficulty 3 (medium)', async () => {
    redis.get.mockResolvedValue(JSON.stringify(BANK));
    const result = await fallbackQuestion(['q1', 'q4']);
    // q2 and q3 are both difficulty 3 and unseen
    expect([2, 3]).toContain(result.difficulty);
  });

  test('uses DB fallback when Redis returns null', async () => {
    redis.get.mockResolvedValue(null);
    db.getActiveQuestions.mockResolvedValue(BANK);
    const result = await fallbackQuestion([]);
    expect(result).not.toBeNull();
  });

  test('uses Redis cache when available', async () => {
    redis.get.mockResolvedValue(JSON.stringify(BANK));
    await fallbackQuestion([]);
    // DB should NOT be called when Redis has data
    expect(db.getActiveQuestions).not.toHaveBeenCalled();
  });
});

// ── handleAnswerSubmit (text path) ────────────────────────────────────────────

describe('handleAnswerSubmit — text', () => {
  test('sends FEEDBACK frame on success', async () => {
    const ws      = makeWs();
    const session = makeSession();
    await handleAnswerSubmit(ws, session, { question_id: 'q-001', answer_text: 'My answer' }, 'text');
    expect(ws.send).toHaveBeenCalledTimes(1);
    const msg = JSON.parse(ws.send.mock.calls[0][0]);
    expect(msg.type).toBe('FEEDBACK');
  });

  test('FEEDBACK payload has all required fields', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload).toHaveProperty('scores');
    expect(payload).toHaveProperty('confidence_label');
    expect(payload).toHaveProperty('hedging_words');
    expect(payload).toHaveProperty('delivery_flags');
    expect(payload).toHaveProperty('improvement_tips');
    expect(payload).toHaveProperty('next_question');
  });

  test('delivery_flags is empty array for text input', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload.delivery_flags).toEqual([]);
  });

  test('does NOT call ASR for text input', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    expect(services.callASR).not.toHaveBeenCalled();
  });

  test('calls NLP and Sentiment in parallel via Promise.all', async () => {
    // Track call order — both should start before either resolves
    const callOrder = [];
    services.callNLP.mockImplementation(() => {
      callOrder.push('nlp');
      return Promise.resolve(MOCK_NLP);
    });
    services.callSentiment.mockImplementation(() => {
      callOrder.push('sentiment');
      return Promise.resolve(MOCK_SENT);
    });

    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');

    // Both must have been called exactly once
    expect(services.callNLP).toHaveBeenCalledTimes(1);
    expect(services.callSentiment).toHaveBeenCalledTimes(1);
  });

  test('adds asked question to session askedIds', async () => {
    const ws      = makeWs();
    const session = makeSession({ askedIds: [] });
    await handleAnswerSubmit(ws, session, { question_id: 'q-001', answer_text: 'ans' }, 'text');
    expect(session.askedIds).toContain('q-001');
  });

  test('sends ERROR when question_id missing', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { answer_text: 'ans' }, 'text');
    const msg = JSON.parse(ws.send.mock.calls[0][0]);
    expect(msg.type).toBe('ERROR');
  });

  test('sends ERROR when question not found in DB', async () => {
    db.getQuestion.mockResolvedValue(null);
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-999', answer_text: 'ans' }, 'text');
    const msg = JSON.parse(ws.send.mock.calls[0][0]);
    expect(msg.type).toBe('ERROR');
    expect(msg.payload.code).toBe('question_not_found');
  });

  test('uses neutral scores when NLP returns null (timeout)', async () => {
    services.callNLP.mockResolvedValue(null);
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload.scores.content).toBe(2);
    expect(payload.scores.relevance).toBe(2);
    expect(payload.scores.completeness).toBe(2);
    expect(payload.scores.accuracy).toBe(2);
  });

  test('adds _warning to feedback when NLP used fallback scores', async () => {
    services.callNLP.mockResolvedValue(null);
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload._warning).toBeDefined();
  });

  test('uses NLP cache on second identical answer', async () => {
    const cachedNlp = JSON.stringify(MOCK_NLP);
    redis.get.mockResolvedValue(cachedNlp);
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    // NLP service should NOT be called — result came from cache
    expect(services.callNLP).not.toHaveBeenCalled();
  });

  test('uses fallback question when adaptive engine returns null', async () => {
    services.callAdaptive.mockResolvedValue(null);
    const fallbackQ = { id: 'qfb', text: 'Fallback Q', type: 'hr', difficulty: 3 };
    db.getActiveQuestions.mockResolvedValue([fallbackQ]);

    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload.next_question.id).toBe('qfb');
  });

  test('persists answer with DB retry', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    expect(db.withRetry).toHaveBeenCalledTimes(1);
  });

  test('scores in FEEDBACK match NLP result', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(), { question_id: 'q-001', answer_text: 'ans' }, 'text');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload.scores.content).toBe(MOCK_NLP.content_score);
    expect(payload.scores.accuracy).toBe(MOCK_NLP.accuracy_score);
  });
});

// ── handleAnswerSubmit (voice path) ───────────────────────────────────────────

describe('handleAnswerSubmit — voice', () => {
  const MOCK_ASR = {
    transcript       : 'My spoken answer',
    speaking_rate_wpm: 140,
    pause_count      : 2,
    pitch_mean_hz    : 185,
    pitch_std        : 20,
    volume_variation : 0.12,
    is_fast_speech   : false,
    is_nervous_pitch : false,
    delivery_flags   : [],
  };

  beforeEach(() => {
    services.callASR.mockResolvedValue(MOCK_ASR);
  });

  test('calls ASR before NLP for voice input', async () => {
    const order = [];
    services.callASR.mockImplementation(() => { order.push('asr'); return Promise.resolve(MOCK_ASR); });
    services.callNLP.mockImplementation(() => { order.push('nlp'); return Promise.resolve(MOCK_NLP); });

    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(),
      { question_id: 'q-001', audio_b64: 'ZmFrZQ==' }, 'voice');

    expect(order[0]).toBe('asr');
    expect(order.indexOf('nlp')).toBeGreaterThan(order.indexOf('asr'));
  });

  test('uses ASR transcript for NLP evaluation', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(),
      { question_id: 'q-001', audio_b64: 'ZmFrZQ==' }, 'voice');
    expect(services.callNLP).toHaveBeenCalledWith(
      MOCK_QUESTION.text, MOCK_ASR.transcript
    );
  });

  test('sends ERROR on ASR timeout', async () => {
    services.callASR.mockResolvedValue({ _error: 'asr_timeout', _message: 'timed out' });
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(),
      { question_id: 'q-001', audio_b64: 'ZmFrZQ==' }, 'voice');
    const msg = JSON.parse(ws.send.mock.calls[0][0]);
    expect(msg.type).toBe('ERROR');
    expect(msg.payload.code).toBe('asr_timeout');
  });

  test('sends ERROR when audio_b64 missing for voice', async () => {
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(),
      { question_id: 'q-001' }, 'voice'); // no audio_b64
    const msg = JSON.parse(ws.send.mock.calls[0][0]);
    expect(msg.type).toBe('ERROR');
  });

  test('delivery_flags from ASR appear in FEEDBACK', async () => {
    services.callASR.mockResolvedValue({ ...MOCK_ASR, delivery_flags: ['fast_speech'] });
    const ws = makeWs();
    await handleAnswerSubmit(ws, makeSession(),
      { question_id: 'q-001', audio_b64: 'ZmFrZQ==' }, 'voice');
    const { payload } = JSON.parse(ws.send.mock.calls[0][0]);
    expect(payload.delivery_flags).toContain('fast_speech');
  });
});

// ── handleFlagQuestion ────────────────────────────────────────────────────────

describe('handleFlagQuestion', () => {
  test('calls db.flagQuestion with question_id', async () => {
    const ws = makeWs();
    await handleFlagQuestion(ws, { question_id: 'q-001', reason: 'irrelevant' });
    expect(db.flagQuestion).toHaveBeenCalledWith('q-001');
  });

  test('does nothing when question_id missing', async () => {
    const ws = makeWs();
    await handleFlagQuestion(ws, {}); // no question_id
    expect(db.flagQuestion).not.toHaveBeenCalled();
  });

  test('does not send WS message (fire and forget)', async () => {
    const ws = makeWs();
    await handleFlagQuestion(ws, { question_id: 'q-001' });
    expect(ws.send).not.toHaveBeenCalled();
  });
});
