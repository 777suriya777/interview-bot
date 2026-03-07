/**
 * server.js — Chatbot Engine: WebSocket orchestrator + HTTP API.
 *
 * HTTP routes:
 *   POST /api/auth/register   — user registration
 *   POST /api/auth/login      — user login
 *   POST /api/sessions        — create session [auth]
 *   GET  /api/sessions/:id/report — get session report [auth]
 *   GET  /api/health          — liveness probe
 *
 * WebSocket:
 *   ws://<host>:3001/session?token=<JWT>
 *   Message types: START_SESSION, SUBMIT_TEXT, SUBMIT_VOICE, FLAG_QUESTION, END_SESSION
 *
 * Architecture note:
 *   HTTP and WebSocket share the same port (3001) via Node's built-in
 *   'upgrade' event. The ws library handles the WebSocket upgrade;
 *   the http module handles normal HTTP requests via a minimal router.
 */

'use strict';

const http    = require('http');
const url     = require('url');
const { WebSocketServer } = require('ws');
const bcrypt  = require('bcryptjs');

const db      = require('./db/queries');
const redis   = require('./cache/redis');
const { signToken, verifyJWT, verifyWsToken } = require('./middleware/auth');
const { handleStartSession, handleEndSession } = require('./handlers/session');
const { handleAnswerSubmit, handleFlagQuestion } = require('./handlers/answer');
const { callReportGenerate } = require('./services');

const PORT = parseInt(process.env.PORT || '3001', 10);
const CORS_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:5173';

// ── Minimal HTTP router ───────────────────────────────────────────────────────

/**
 * Parse JSON body from an IncomingMessage.
 * @returns {Promise<object>}
 */
function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(data || '{}')); }
      catch { reject(new Error('Invalid JSON body')); }
    });
    req.on('error', reject);
  });
}

function sendJSON(res, statusCode, body) {
  const payload = JSON.stringify(body);
  res.writeHead(statusCode, {
    'Content-Type'                : 'application/json',
    'Content-Length'              : Buffer.byteLength(payload),
    'Access-Control-Allow-Origin' : CORS_ORIGIN,
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  });
  res.end(payload);
}

// ── Auth routes ───────────────────────────────────────────────────────────────

async function handleRegister(req, res) {
  let body;
  try { body = await readBody(req); }
  catch { return sendJSON(res, 400, { error: 'invalid_json', detail: 'Request body must be JSON' }); }

  const { email, name, password } = body;
  if (!email || !name || !password) {
    return sendJSON(res, 400, { error: 'missing_fields', detail: 'email, name, and password are required' });
  }

  // Check duplicate email
  const existing = await db.getUserByEmail(email).catch(() => null);
  if (existing) {
    return sendJSON(res, 409, { error: 'email_exists', detail: 'This email is already registered' });
  }

  const hashedPassword = await bcrypt.hash(password, 12);

  try {
    const user         = await db.createUser({ email, name, hashedPassword });
    const access_token = signToken(user.id, user.email);
    return sendJSON(res, 201, { user_id: user.id, access_token, token_type: 'bearer' });
  } catch (err) {
    console.error('[Auth] Register error:', err.message);
    return sendJSON(res, 500, { error: 'server_error', detail: 'Registration failed' });
  }
}

async function handleLogin(req, res) {
  let body;
  try { body = await readBody(req); }
  catch { return sendJSON(res, 400, { error: 'invalid_json', detail: 'Request body must be JSON' }); }

  const { email, password } = body;
  if (!email || !password) {
    return sendJSON(res, 400, { error: 'missing_fields', detail: 'email and password are required' });
  }

  const user = await db.getUserByEmail(email).catch(() => null);
  if (!user || !user.hashed_password) {
    return sendJSON(res, 401, { error: 'invalid_credentials', detail: 'Invalid email or password' });
  }

  const valid = await bcrypt.compare(password, user.hashed_password);
  if (!valid) {
    return sendJSON(res, 401, { error: 'invalid_credentials', detail: 'Invalid email or password' });
  }

  const access_token = signToken(user.id, user.email);
  const expires_in   = parseInt(process.env.JWT_EXPIRES_IN || '86400', 10);
  return sendJSON(res, 200, { user_id: user.id, access_token, expires_in });
}

// ── Session HTTP routes ───────────────────────────────────────────────────────

async function handleCreateSession(req, res, user) {
  let body;
  try { body = await readBody(req); }
  catch { return sendJSON(res, 400, { error: 'invalid_json', detail: 'Request body must be JSON' }); }

  const { interview_type, target_role, duration_minutes } = body;
  const validTypes = ['technical', 'behavioural', 'hr', 'mixed'];
  if (!validTypes.includes(interview_type)) {
    return sendJSON(res, 400, {
      error  : 'invalid_type',
      detail : `interview_type must be one of: ${validTypes.join(', ')}`,
    });
  }

  try {
    const session = await db.createSession({
      userId        : user.userId,
      interviewType : interview_type,
      targetRole    : target_role,
      difficultyLevel: 3, // always start at medium difficulty
    });

    // Select the first question
    const firstQuestion = await db.getFirstQuestion(interview_type);
    if (!firstQuestion) {
      return sendJSON(res, 503, { error: 'no_questions', detail: 'Question bank is empty' });
    }

    return sendJSON(res, 201, {
      session_id     : session.id,
      first_question : {
        id         : firstQuestion.id,
        text       : firstQuestion.text,
        type       : firstQuestion.type,
        difficulty : firstQuestion.difficulty,
      },
    });
  } catch (err) {
    console.error('[Session] Create error:', err.message);
    return sendJSON(res, 500, { error: 'server_error', detail: 'Session creation failed' });
  }
}

async function handleGetReport(req, res, user, sessionId) {
  try {
    const session = await db.getSession(sessionId);
    if (!session) {
      return sendJSON(res, 404, { error: 'not_found', detail: 'Session not found' });
    }
    if (session.user_id.toString() !== user.userId) {
      return sendJSON(res, 403, { error: 'forbidden', detail: 'Access denied' });
    }

    const report = await callReportGenerate(sessionId);
    if (!report) {
      return sendJSON(res, 503, { error: 'report_unavailable', detail: 'Report service unavailable' });
    }

    return sendJSON(res, 200, report);
  } catch (err) {
    console.error('[Report] Get error:', err.message);
    return sendJSON(res, 500, { error: 'server_error', detail: 'Failed to retrieve report' });
  }
}

async function handleGetSessions(req, res, user) {
  try {
    const sessions = await db.getSessionsByUser(user.userId, 20);
    return sendJSON(res, 200, { sessions });
  } catch (err) {
    console.error('[Sessions] List error:', err.message);
    return sendJSON(res, 500, { error: 'server_error', detail: 'Failed to retrieve sessions' });
  }
}

async function handleGetPerformance(req, res, user) {
  try {
    const rows = await db.getUserPerformance(user.userId);

    const topics = {};
    for (const row of rows) {
      topics[row.topic] = {
        score:        parseFloat(row.score),
        answer_count: row.answer_count,
      };
    }

    const scores  = Object.values(topics).map(t => t.score);
    const overall = scores.length
      ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 10) / 10
      : 0;

    // Priority areas = topics below average, sorted weakest first, max 3
    const avg            = overall;
    const priority_areas = rows
      .filter(r => parseFloat(r.score) < avg)
      .slice(0, 3)
      .map(r => r.topic);

    return sendJSON(res, 200, { overall_score: overall, topics, priority_areas });
  } catch (err) {
    console.error('[Performance] Get error:', err.message);
    return sendJSON(res, 500, { error: 'server_error', detail: 'Failed to retrieve performance' });
  }
}

// ── HTTP request handler ──────────────────────────────────────────────────────

async function requestHandler(req, res) {
  const parsed  = url.parse(req.url, true);
  const path    = parsed.pathname;
  const method  = req.method.toUpperCase();

  // CORS preflight
  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin' : CORS_ORIGIN,
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    });
    return res.end();
  }

  // Health check (no auth)
  if (path === '/api/health' && method === 'GET') {
    const redisOk = await redis.ping();
    return sendJSON(res, 200, { status: 'ok', redis: redisOk });
  }

  // Auth routes (no JWT required)
  if (path === '/api/auth/register' && method === 'POST') return handleRegister(req, res);
  if (path === '/api/auth/login'    && method === 'POST') return handleLogin(req, res);

  // ── Protected routes — verify JWT before proceeding ──────────────────────
  const authHeader = req.headers['authorization'] || '';
  const token      = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
  if (!token) {
    return sendJSON(res, 401, { error: 'missing_token', detail: 'Authorization header required' });
  }
  let user;
  try {
    user = verifyWsToken(token); // reuse the same verify function
  } catch {
    return sendJSON(res, 401, { error: 'invalid_token', detail: 'Invalid or expired token' });
  }

  // POST /api/sessions
  if (path === '/api/sessions' && method === 'POST') {
    return handleCreateSession(req, res, user);
  }

  // GET /api/sessions — session history list
  if (path === '/api/sessions' && method === 'GET') {
    return handleGetSessions(req, res, user);
  }

  // GET /api/performance — user topic performance
  if (path === '/api/performance' && method === 'GET') {
    return handleGetPerformance(req, res, user);
  }

  // GET /api/sessions/:id/report
  const reportMatch = path.match(/^\/api\/sessions\/([^/]+)\/report$/);
  if (reportMatch && method === 'GET') {
    return handleGetReport(req, res, user, reportMatch[1]);
  }

  return sendJSON(res, 404, { error: 'not_found', detail: `${method} ${path} not found` });
}

// ── WebSocket server ──────────────────────────────────────────────────────────

const httpServer = http.createServer(requestHandler);
const wss        = new WebSocketServer({ noServer: true });

/**
 * Authenticate WebSocket upgrade requests.
 * Token is read from the query string: ws://host:3001/session?token=<JWT>
 * Returns 401 and destroys the socket if token is invalid.
 */
httpServer.on('upgrade', (req, socket, head) => {
  const { pathname, query } = url.parse(req.url, true);

  if (pathname !== '/session') {
    socket.write('HTTP/1.1 404 Not Found\r\n\r\n');
    socket.destroy();
    return;
  }

  let user;
  try {
    user = verifyWsToken(query.token);
  } catch {
    socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
    socket.destroy();
    return;
  }

  wss.handleUpgrade(req, socket, head, (ws) => {
    ws.user = user; // attach decoded JWT payload to ws instance
    wss.emit('connection', ws, req);
  });
});

// ── WebSocket message routing ─────────────────────────────────────────────────

wss.on('connection', (ws) => {
  const userId = ws.user?.userId || 'unknown';
  console.log(`[WS] Client connected: user ${userId}`);

  ws.on('message', async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      ws.send(JSON.stringify({ type: 'ERROR', payload: { code: 'invalid_json', message: 'Message must be JSON' } }));
      return;
    }

    const { type, payload = {} } = msg;

    // Route to the appropriate handler
    try {
      switch (type) {
        case 'START_SESSION':
          await handleStartSession(ws, payload, ws.user);
          break;

        case 'SUBMIT_TEXT':
          if (!ws.sessionState) {
            ws.send(JSON.stringify({ type: 'ERROR', payload: { code: 'no_session', message: 'Send START_SESSION first' } }));
          } else {
            await handleAnswerSubmit(ws, ws.sessionState, payload, 'text');
          }
          break;

        case 'SUBMIT_VOICE':
          if (!ws.sessionState) {
            ws.send(JSON.stringify({ type: 'ERROR', payload: { code: 'no_session', message: 'Send START_SESSION first' } }));
          } else {
            await handleAnswerSubmit(ws, ws.sessionState, payload, 'voice');
          }
          break;

        case 'FLAG_QUESTION':
          await handleFlagQuestion(ws, payload);
          break;

        case 'END_SESSION':
          await handleEndSession(ws, ws.user);
          break;

        default:
          ws.send(JSON.stringify({ type: 'ERROR', payload: { code: 'unknown_type', message: `Unknown message type: ${type}` } }));
      }
    } catch (err) {
      console.error(`[WS] Unhandled error processing ${type}:`, err.message);
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ type: 'ERROR', payload: { code: 'server_error', message: 'An unexpected error occurred' } }));
      }
    }
  });

  ws.on('close', async () => {
    console.log(`[WS] Client disconnected: user ${userId}`);
    // Session state is already persisted to Redis — user can reconnect and resume
    // If session was active, mark as abandoned after a grace period
    if (ws.sessionState) {
      const { sessionId } = ws.sessionState;
      // Wait 5 minutes before marking abandoned (give user time to reconnect)
      setTimeout(async () => {
        try {
          await db.abandonSession(sessionId);
        } catch { /* session may have been completed normally */ }
      }, 5 * 60 * 1000);
    }
  });

  ws.on('error', (err) => {
    console.error(`[WS] Socket error (user ${userId}):`, err.message);
  });
});

// ── Startup ───────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[Server] Chatbot Engine listening on port ${PORT}`);
  console.log(`[Server] WebSocket endpoint: ws://localhost:${PORT}/session?token=<JWT>`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('[Server] SIGTERM received — shutting down gracefully');
  httpServer.close(() => {
    db.pool.end().then(() => process.exit(0));
  });
});

module.exports = { httpServer, wss }; // exported for tests
