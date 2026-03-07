/**
 * __tests__/auth.test.js
 * Tests for middleware/auth.js — JWT signing and verification.
 */

'use strict';

process.env.JWT_SECRET     = 'test_secret_for_auth_tests';
process.env.JWT_EXPIRES_IN = '3600';

const { signToken, verifyToken, verifyWsToken } = require('../middleware/auth');

describe('signToken / verifyToken', () => {
  test('signToken returns a string', () => {
    const token = signToken('user-1', 'test@example.com');
    expect(typeof token).toBe('string');
    expect(token.split('.')).toHaveLength(3); // JWT has 3 parts
  });

  test('verifyToken decodes userId and email', () => {
    const token   = signToken('user-123', 'alice@example.com');
    const decoded = verifyToken(token);
    expect(decoded.userId).toBe('user-123');
    expect(decoded.email).toBe('alice@example.com');
  });

  test('verifyToken throws on tampered token', () => {
    const token    = signToken('user-1', 'a@b.com');
    const tampered = token.slice(0, -5) + 'XXXXX';
    expect(() => verifyToken(tampered)).toThrow();
  });

  test('verifyToken throws on wrong secret', () => {
    const jwt   = require('jsonwebtoken');
    const other = jwt.sign({ userId: 'x' }, 'wrong_secret', { expiresIn: 3600 });
    expect(() => verifyToken(other)).toThrow();
  });

  test('verifyToken throws TokenExpiredError on expired token', () => {
    const jwt       = require('jsonwebtoken');
    const expired   = jwt.sign({ userId: 'x', email: 'x@x.com' }, 'test_secret_for_auth_tests', { expiresIn: -1 });
    expect(() => verifyToken(expired)).toThrow(/expired/i);
  });
});

describe('verifyWsToken', () => {
  test('returns decoded payload for valid token', () => {
    const token   = signToken('ws-user', 'ws@example.com');
    const decoded = verifyWsToken(token);
    expect(decoded.userId).toBe('ws-user');
  });

  test('throws with statusCode 401 for null token', () => {
    expect(() => verifyWsToken(null)).toThrow();
    try { verifyWsToken(null); }
    catch (e) { expect(e.statusCode).toBe(401); }
  });

  test('throws with statusCode 401 for invalid token', () => {
    try { verifyWsToken('not.a.jwt'); }
    catch (e) { expect(e.statusCode).toBe(401); }
  });
});
