/**
 * cache/redis.js — Thin wrapper around ioredis.
 *
 * Exports a Redis client instance that silently degrades when Redis is
 * unavailable — all operations resolve to null/false rather than throwing,
 * so the application continues without caching.
 *
 * The wrapper exposes the same get/set/del/exists interface as ioredis
 * but swallows connection errors at the method level.
 */

const Redis = require('ioredis');

const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379/0';

const client = new Redis(REDIS_URL, {
  maxRetriesPerRequest    : 2,
  enableReadyCheck        : false,
  lazyConnect             : true,  // don't connect until first command
  reconnectOnError        : () => true,
});

client.on('error', (err) => {
  // Suppress repetitive connection error spam
  if (!client._lastErrMsg || client._lastErrMsg !== err.message) {
    console.warn(`[Redis] Connection error: ${err.message}`);
    client._lastErrMsg = err.message;
  }
});

client.on('connect', () => {
  console.log('[Redis] Connected');
  client._lastErrMsg = null;
});

/**
 * GET with silent failure.
 * @returns {string|null}
 */
async function get(key) {
  try {
    return await client.get(key);
  } catch {
    return null;
  }
}

/**
 * SET with optional expiry args (e.g. 'EX', 3600).
 * @returns {string|null} 'OK' on success, null on error
 */
async function set(key, value, ...expiryArgs) {
  try {
    return await client.set(key, value, ...expiryArgs);
  } catch {
    return null;
  }
}

/**
 * DEL one or more keys.
 */
async function del(...keys) {
  try {
    return await client.del(...keys);
  } catch {
    return 0;
  }
}

/**
 * PING — used to check connectivity.
 * @returns {boolean}
 */
async function ping() {
  try {
    const res = await client.ping();
    return res === 'PONG';
  } catch {
    return false;
  }
}

module.exports = { client, get, set, del, ping };
