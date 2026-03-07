/**
 * middleware/auth.js — JWT authentication helpers.
 *
 * Used for:
 *  - HTTP endpoints: verifyJWT middleware (Express-style)
 *  - WebSocket upgrade: verifyWsToken(token) → decoded payload or throws
 *
 * Token shape: { userId: string, email: string, iat: number, exp: number }
 */

const jwt = require('jsonwebtoken');

const JWT_SECRET     = process.env.JWT_SECRET || 'dev_secret_change_in_prod';
const JWT_EXPIRES_IN = parseInt(process.env.JWT_EXPIRES_IN || '86400', 10);

/**
 * Sign a JWT for a user.
 *
 * @param {string} userId
 * @param {string} email
 * @returns {string} signed JWT
 */
function signToken(userId, email) {
  return jwt.sign(
    { userId, email },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

/**
 * Verify a JWT token string and return the decoded payload.
 * Throws jwt.JsonWebTokenError or jwt.TokenExpiredError on failure.
 *
 * @param {string} token
 * @returns {{ userId: string, email: string, iat: number, exp: number }}
 */
function verifyToken(token) {
  return jwt.verify(token, JWT_SECRET);
}

/**
 * Express-style middleware.
 * Reads Bearer token from Authorization header and attaches decoded payload
 * to req.user. Responds 401 if missing or invalid.
 */
function verifyJWT(req, res, next) {
  const header = req.headers['authorization'] || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ error: 'missing_token', detail: 'Authorization header required' });
  }

  try {
    req.user = verifyToken(token);
    next();
  } catch (err) {
    const code = err.name === 'TokenExpiredError' ? 'token_expired' : 'invalid_token';
    return res.status(401).json({ error: code, detail: err.message });
  }
}

/**
 * Verify a WebSocket upgrade token (from query string: ?token=<JWT>).
 * Used in the 'upgrade' event handler before the WS connection is accepted.
 *
 * @param {string|null} token
 * @returns {{ userId: string, email: string }}
 * @throws {Error} with .statusCode = 401 on failure
 */
function verifyWsToken(token) {
  if (!token) {
    const err = new Error('Missing token');
    err.statusCode = 401;
    throw err;
  }
  try {
    return verifyToken(token);
  } catch (e) {
    const err = new Error(e.message);
    err.statusCode = 401;
    throw err;
  }
}

module.exports = { signToken, verifyToken, verifyJWT, verifyWsToken };
