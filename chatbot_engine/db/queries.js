/**
 * db/queries.js — All PostgreSQL queries for the chatbot engine.
 *
 * Uses a connection pool (pg.Pool) shared across the process.
 * All UUIDs are cast explicitly in SQL to avoid type mismatch errors.
 * All timestamps are stored in UTC (PostgreSQL TIMESTAMPTZ).
 */

const { Pool } = require('pg');

// ── Connection pool ───────────────────────────────────────────────────────────
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 5_000,
});

pool.on('error', (err) => {
  console.error('[DB] Unexpected pool error:', err.message);
});

// ── User queries ──────────────────────────────────────────────────────────────

/**
 * Find a user by email. Returns the row or null.
 */
async function getUserByEmail(email) {
  const { rows } = await pool.query(
    'SELECT id, email, name, hashed_password, oauth_provider FROM users WHERE email = $1',
    [email]
  );
  return rows[0] || null;
}

/**
 * Create a new user. Returns the created row.
 */
async function createUser({ email, name, hashedPassword, oauthProvider, oauthSub }) {
  const { rows } = await pool.query(
    `INSERT INTO users (email, name, hashed_password, oauth_provider, oauth_sub)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING id, email, name`,
    [email, name, hashedPassword || null, oauthProvider || null, oauthSub || null]
  );
  return rows[0];
}

// ── Session queries ───────────────────────────────────────────────────────────

/**
 * Create a new session. Returns the created row.
 */
async function createSession({ userId, interviewType, targetRole, difficultyLevel }) {
  const { rows } = await pool.query(
    `INSERT INTO sessions (user_id, interview_type, target_role, difficulty_level)
     VALUES ($1::uuid, $2, $3, $4)
     RETURNING id, user_id, interview_type, target_role, difficulty_level, status, started_at`,
    [userId, interviewType, targetRole || null, difficultyLevel || 3]
  );
  return rows[0];
}

/**
 * Load a session by ID. Returns the row or null.
 */
async function getSession(sessionId) {
  const { rows } = await pool.query(
    `SELECT id, user_id, interview_type, target_role, difficulty_level,
            status, performance_score, started_at, ended_at
     FROM sessions
     WHERE id = $1::uuid`,
    [sessionId]
  );
  return rows[0] || null;
}

/**
 * Mark a session as completed and record the end time.
 */
async function completeSession(sessionId, reportUrl) {
  await pool.query(
    `UPDATE sessions
     SET status = 'completed', ended_at = NOW(), report_url = $2
     WHERE id = $1::uuid`,
    [sessionId, reportUrl || null]
  );
}

/**
 * Mark a session as abandoned (user disconnected without finishing).
 */
async function abandonSession(sessionId) {
  await pool.query(
    `UPDATE sessions
     SET status = 'abandoned', ended_at = NOW()
     WHERE id = $1::uuid AND status = 'active'`,
    [sessionId]
  );
}

// ── Question queries ──────────────────────────────────────────────────────────

/**
 * Load a single question by ID.
 */
async function getQuestion(questionId) {
  const { rows } = await pool.query(
    `SELECT id, text, type, difficulty, category, key_concepts
     FROM questions
     WHERE id = $1::uuid AND is_active = TRUE`,
    [questionId]
  );
  return rows[0] || null;
}

/**
 * Load the first question for a new session.
 * Selects a random question at difficulty 3 (medium start).
 */
async function getFirstQuestion(interviewType) {
  // For 'mixed' sessions, any type is valid; otherwise filter by type
  const typeFilter = interviewType === 'mixed' ? '' : 'AND type = $2';
  const params = interviewType === 'mixed' ? [3] : [3, interviewType];

  const { rows } = await pool.query(
    `SELECT id, text, type, difficulty, category
     FROM questions
     WHERE difficulty = $1 AND is_active = TRUE ${typeFilter}
     ORDER BY RANDOM()
     LIMIT 1`,
    params
  );
  return rows[0] || null;
}

/**
 * Load all active questions (used as fallback when adaptive engine is unavailable).
 * Returns lightweight objects — id, type, difficulty only.
 */
async function getActiveQuestions() {
  const { rows } = await pool.query(
    `SELECT id::text, text, type, difficulty, COALESCE(category, '') AS category
     FROM questions
     WHERE is_active = TRUE
     ORDER BY created_at DESC`
  );
  return rows;
}

/**
 * Increment the flag_count on a question (for quality review).
 */
async function flagQuestion(questionId) {
  await pool.query(
    `UPDATE questions
     SET flag_count = flag_count + 1
     WHERE id = $1::uuid`,
    [questionId]
  );
}

// ── Answer queries ────────────────────────────────────────────────────────────

/**
 * Insert a completed answer record.
 * Returns the created answer ID.
 *
 * @param {object} answer
 * @param {string} answer.sessionId
 * @param {string} answer.questionId
 * @param {string} answer.inputMode          'text' | 'voice'
 * @param {string} [answer.transcript]
 * @param {number} answer.contentScore       0-4
 * @param {number} answer.relevanceScore     0-4
 * @param {number} answer.completenessScore  0-4
 * @param {number} answer.accuracyScore      0-4
 * @param {number} answer.overallScore       0.0-4.0
 * @param {string} answer.confidenceLabel    'high'|'moderate'|'low'|'anxious'
 * @param {Array}  answer.hedgingFlags       [{word, position}]
 * @param {Array}  answer.deliveryFlags      string[]
 * @param {number} [answer.speakingRateWpm]
 * @param {number} [answer.pitchStd]
 */
async function insertAnswer(answer) {
  const {
    sessionId, questionId, inputMode, transcript,
    contentScore, relevanceScore, completenessScore, accuracyScore, overallScore,
    confidenceLabel, hedgingFlags, deliveryFlags,
    speakingRateWpm, pitchStd,
  } = answer;

  const { rows } = await pool.query(
    `INSERT INTO answers (
       session_id, question_id, input_mode, transcript,
       content_score, relevance_score, completeness_score, accuracy_score, overall_score,
       confidence_label, hedging_flags, delivery_flags,
       speaking_rate_wpm, pitch_std
     ) VALUES (
       $1::uuid, $2::uuid, $3, $4,
       $5, $6, $7, $8, $9,
       $10, $11::jsonb, $12::jsonb,
       $13, $14
     )
     RETURNING id`,
    [
      sessionId, questionId, inputMode, transcript || null,
      contentScore, relevanceScore, completenessScore, accuracyScore, overallScore,
      confidenceLabel,
      JSON.stringify(hedgingFlags || []),
      JSON.stringify(deliveryFlags || []),
      speakingRateWpm || null, pitchStd || null,
    ]
  );
  return rows[0].id;
}

// ── Retry wrapper ─────────────────────────────────────────────────────────────

/**
 * Retry a DB operation up to maxRetries times with exponential backoff.
 * Used for answer inserts — we never want to silently drop an answer.
 *
 * @param {Function} fn          async function to retry
 * @param {number}   maxRetries  default 3 (spec: retry 3x on DB write failure)
 * @param {number}   baseDelayMs initial delay in ms (doubles each attempt)
 */
/**
 * Get a user's session history, most recent first.
 * Returns lightweight rows for the dashboard list (no answer details).
 */
async function getSessionsByUser(userId, limit = 20) {
  const { rows } = await pool.query(
    `SELECT
       id::text,
       interview_type,
       status,
       target_role,
       difficulty_level,
       performance_score,
       started_at,
       ended_at,
       report_url
     FROM sessions
     WHERE user_id = $1::uuid
     ORDER BY started_at DESC
     LIMIT $2`,
    [userId, limit]
  );
  return rows;
}

/**
 * Get per-topic performance summary for a user.
 * Returns rows ordered by score ascending (weakest first).
 */
async function getUserPerformance(userId) {
  const { rows } = await pool.query(
    `SELECT topic, score, answer_count, updated_at
     FROM user_performance
     WHERE user_id = $1::uuid
     ORDER BY score ASC`,
    [userId]
  );
  return rows;
}

async function withRetry(fn, maxRetries = 3, baseDelayMs = 500) {
  let lastError;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        const delay = baseDelayMs * Math.pow(2, attempt - 1); // 500, 1000, 2000
        console.warn(`[DB] Retry ${attempt}/${maxRetries} after ${delay}ms: ${err.message}`);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError;
}

module.exports = {
  pool,
  getUserByEmail,
  createUser,
  createSession,
  getSession,
  completeSession,
  abandonSession,
  getQuestion,
  getFirstQuestion,
  getActiveQuestions,
  flagQuestion,
  insertAnswer,
  getSessionsByUser,
  getUserPerformance,
  withRetry,
};
