/**
 * handlers/session.js — WebSocket session lifecycle management.
 *
 * Handles:
 *  START_SESSION  — validate session belongs to user, load Redis state or init fresh
 *  END_SESSION    — mark session complete, trigger report generation
 *  reconnect      — restore session context from Redis on reconnect (TTL 24h)
 */

const db      = require('../db/queries');
const redis   = require('../cache/redis');
const { callReportGenerate } = require('../services');

// ── Session state helpers ─────────────────────────────────────────────────────

/**
 * Load session state from Redis, or initialise a fresh state from DB.
 *
 * Redis key: session:{sessionId}:state  TTL 24h
 *
 * @param {string} sessionId
 * @param {string} userId
 * @returns {object} session state { sessionId, userId, askedIds, difficulty, performance }
 */
async function loadSessionState(sessionId, userId) {
  const key = `session:${sessionId}:state`;

  // Try Redis first (hot path — session already in progress)
  const cached = await redis.get(key);
  if (cached) {
    try {
      const state = JSON.parse(cached);
      // Verify the session belongs to this user (prevent session hijacking)
      if (state.userId !== userId) {
        throw new Error('Session user mismatch');
      }
      return state;
    } catch (err) {
      console.warn(`[Session] Redis state parse failed (${sessionId}): ${err.message}`);
    }
  }

  // Fall back to DB — first question of a new session or Redis expired
  const session = await db.getSession(sessionId);
  if (!session) {
    throw new Error(`Session ${sessionId} not found`);
  }
  if (session.user_id.toString() !== userId) {
    throw new Error('Session does not belong to this user');
  }
  if (session.status !== 'active') {
    throw new Error(`Session is ${session.status}, not active`);
  }

  const freshState = {
    sessionId,
    userId,
    resumeText  : session.resume_text || null,
    askedIds    : [],
    difficulty  : session.difficulty_level || 3,
    performance : parseFloat(session.performance_score) || 0,
  };

  await saveSessionState(freshState);
  return freshState;
}

/**
 * Persist session state to Redis with 24h TTL.
 *
 * @param {object} state
 */
async function saveSessionState(state) {
  const key = `session:${state.sessionId}:state`;
  await redis.set(key, JSON.stringify(state), 'EX', 86_400);
}

/**
 * Delete session state from Redis (after session ends).
 *
 * @param {string} sessionId
 */
async function clearSessionState(sessionId) {
  await redis.del(`session:${sessionId}:state`);
}

// ── Message handlers ──────────────────────────────────────────────────────────

/**
 * Handle START_SESSION message.
 * Validates the session and loads/initialises state.
 * Attaches state to the ws object so answer.js can read it.
 *
 * @param {WebSocket} ws
 * @param {object}    payload  { session_id: string }
 * @param {object}    user     decoded JWT { userId, email }
 */
async function handleStartSession(ws, payload, user) {
  const { session_id: sessionId } = payload;

  if (!sessionId) {
    sendError(ws, 'invalid_payload', 'session_id is required');
    return;
  }

  try {
    const state = await loadSessionState(sessionId, user.userId);
    // Attach to ws so subsequent messages can access it without re-fetching
    ws.sessionState = state;
    console.log(`[Session] Started: ${sessionId} for user ${user.userId}`);
  } catch (err) {
    console.error(`[Session] START_SESSION error: ${err.message}`);
    sendError(ws, 'session_error', err.message);
  }
}

/**
 * Handle END_SESSION message.
 * Marks session complete in DB, triggers report generation, sends SESSION_ENDED.
 *
 * @param {WebSocket} ws
 * @param {object}    user    decoded JWT { userId }
 */
async function handleEndSession(ws, user) {
  const state = ws.sessionState;
  if (!state) {
    sendError(ws, 'no_active_session', 'No session is active on this connection');
    return;
  }

  const { sessionId } = state;
  console.log(`[Session] Ending session ${sessionId}`);

  try {
    // Trigger report generation asynchronously — don't block the WS response
    const reportResult = await callReportGenerate(sessionId);
    const reportUrl = reportResult?.report_pdf_url || '';

    // Persist completion to DB
    await db.completeSession(sessionId, reportUrl);

    // Clean up Redis state
    await clearSessionState(sessionId);
    ws.sessionState = null;

    // Notify client
    ws.send(JSON.stringify({
      type    : 'SESSION_ENDED',
      payload : { session_id: sessionId, report_url: reportUrl },
    }));

    console.log(`[Session] Ended: ${sessionId}. Report: ${reportUrl || '(pending)'}`);
  } catch (err) {
    console.error(`[Session] END_SESSION error: ${err.message}`);
    sendError(ws, 'session_end_error', err.message);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function sendError(ws, code, message) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify({
      type    : 'ERROR',
      payload : { code, message },
    }));
  }
}

module.exports = {
  handleStartSession,
  handleEndSession,
  loadSessionState,
  saveSessionState,
  clearSessionState,
};
