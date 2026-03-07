/**
 * __tests__/db.test.js
 * Tests for db/queries.js — retry logic and query wrappers.
 * The pg Pool is mocked to avoid needing a real database.
 */

'use strict';

jest.mock('pg', () => {
  const mockQuery = jest.fn();
  const Pool      = jest.fn(() => ({
    query : mockQuery,
    on    : jest.fn(),
    end   : jest.fn().mockResolvedValue(undefined),
    _mockQuery: mockQuery,
  }));
  return { Pool };
});

const { Pool }   = require('pg');
const db         = require('../db/queries');
const mockPool   = Pool.mock.results[0].value;

beforeEach(() => {
  mockPool._mockQuery.mockReset();
});

// ── withRetry ─────────────────────────────────────────────────────────────────

describe('withRetry', () => {
  test('succeeds on first attempt', async () => {
    const fn = jest.fn().mockResolvedValue('result');
    const result = await db.withRetry(fn);
    expect(result).toBe('result');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('retries on failure and succeeds on second attempt', async () => {
    const fn = jest.fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValue('ok');
    const result = await db.withRetry(fn, 3, 0); // 0ms delay for speed
    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(2);
  });

  test('throws after maxRetries exhausted', async () => {
    const fn = jest.fn().mockRejectedValue(new Error('persistent failure'));
    await expect(db.withRetry(fn, 3, 0)).rejects.toThrow('persistent failure');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('uses exponential backoff delays', async () => {
    jest.useFakeTimers();
    const fn = jest.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValue('ok');

    const promise = db.withRetry(fn, 3, 100);
    // First retry: 100ms delay
    await jest.advanceTimersByTimeAsync(100);
    // Second retry: 200ms delay
    await jest.advanceTimersByTimeAsync(200);
    const result = await promise;
    expect(result).toBe('ok');
    jest.useRealTimers();
  });

  test('default maxRetries is 3', async () => {
    const fn = jest.fn().mockRejectedValue(new Error('fail'));
    await expect(db.withRetry(fn, undefined, 0)).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(3);
  });
});

// ── getUserByEmail ─────────────────────────────────────────────────────────────

describe('getUserByEmail', () => {
  test('returns user row when found', async () => {
    const user = { id: 'u1', email: 'a@b.com', name: 'Alice' };
    mockPool._mockQuery.mockResolvedValue({ rows: [user] });
    const result = await db.getUserByEmail('a@b.com');
    expect(result).toEqual(user);
  });

  test('returns null when not found', async () => {
    mockPool._mockQuery.mockResolvedValue({ rows: [] });
    const result = await db.getUserByEmail('notfound@x.com');
    expect(result).toBeNull();
  });
});

// ── createSession ─────────────────────────────────────────────────────────────

describe('createSession', () => {
  test('returns created session row', async () => {
    const session = { id: 's1', interview_type: 'technical' };
    mockPool._mockQuery.mockResolvedValue({ rows: [session] });
    const result = await db.createSession({
      userId: 'u1', interviewType: 'technical',
      targetRole: 'SWE', difficultyLevel: 3,
    });
    expect(result).toEqual(session);
  });
});

// ── insertAnswer ──────────────────────────────────────────────────────────────

describe('insertAnswer', () => {
  test('returns created answer id', async () => {
    mockPool._mockQuery.mockResolvedValue({ rows: [{ id: 'ans-1' }] });
    const id = await db.insertAnswer({
      sessionId: 's1', questionId: 'q1', inputMode: 'text',
      transcript: 'My answer', contentScore: 3, relevanceScore: 3,
      completenessScore: 2, accuracyScore: 3, overallScore: 2.75,
      confidenceLabel: 'high', hedgingFlags: [], deliveryFlags: [],
    });
    expect(id).toBe('ans-1');
  });

  test('serialises hedgingFlags and deliveryFlags as JSON', async () => {
    mockPool._mockQuery.mockResolvedValue({ rows: [{ id: 'ans-2' }] });
    const hedging  = [{ word: 'maybe', position: 5 }];
    const delivery = ['fast_speech'];
    await db.insertAnswer({
      sessionId: 's1', questionId: 'q1', inputMode: 'voice',
      transcript: 'text', contentScore: 2, relevanceScore: 2,
      completenessScore: 2, accuracyScore: 2, overallScore: 2.0,
      confidenceLabel: 'moderate', hedgingFlags: hedging, deliveryFlags: delivery,
    });
    const callArgs = mockPool._mockQuery.mock.calls[0][1];
    // hedgingFlags and deliveryFlags should be JSON strings in the query params
    expect(callArgs).toContain(JSON.stringify(hedging));
    expect(callArgs).toContain(JSON.stringify(delivery));
  });
});
