/**
 * useWebSocket.js
 *
 * Custom hook that manages the WebSocket connection lifecycle for an
 * interview session. Handles:
 *   - Initial connection with JWT auth (?token=<jwt> in URL)
 *   - Sending START_SESSION immediately after connect
 *   - Dispatching incoming frames (FEEDBACK, SESSION_ENDED, ERROR) to callbacks
 *   - Automatic reconnect with exponential backoff (up to 4 attempts)
 *   - Restoring last question from sessionStorage on reconnect
 *   - Clean teardown on unmount or session end
 *
 * Usage:
 *   const { send, status, lastError } = useWebSocket({
 *     sessionId, token, onFeedback, onSessionEnded, onError
 *   });
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:3001';
const MAX_RETRIES   = 4;
const BASE_DELAY_MS = 800;

/**
 * @param {object} options
 * @param {string}   options.sessionId      — UUID of the session to join
 * @param {string}   options.token          — JWT access token
 * @param {function} options.onFeedback     — called with FEEDBACK payload
 * @param {function} options.onSessionEnded — called with SESSION_ENDED payload
 * @param {function} options.onError        — called with ERROR payload
 * @param {boolean}  [options.enabled=true] — set false to skip connecting
 */
export function useWebSocket({
  sessionId,
  token,
  onFeedback,
  onSessionEnded,
  onError,
  enabled = true,
}) {
  // 'connecting' | 'open' | 'reconnecting' | 'closed' | 'error'
  const [status,    setStatus]    = useState('connecting');
  const [lastError, setLastError] = useState(null);

  const wsRef        = useRef(null);
  const retriesRef   = useRef(0);
  const timerRef     = useRef(null);
  const unmountedRef = useRef(false);

  // Keep callbacks in refs so effects don't re-run when they change
  const onFeedbackRef     = useRef(onFeedback);
  const onSessionEndedRef = useRef(onSessionEnded);
  const onErrorRef        = useRef(onError);
  useEffect(() => { onFeedbackRef.current     = onFeedback;     }, [onFeedback]);
  useEffect(() => { onSessionEndedRef.current = onSessionEnded; }, [onSessionEnded]);
  useEffect(() => { onErrorRef.current        = onError;        }, [onError]);

  const connect = useCallback(() => {
    if (!enabled || !sessionId || !token) return;

    const url = `${WS_BASE}/session?token=${encodeURIComponent(token)}`;
    const ws  = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return; }
      retriesRef.current = 0;
      setStatus('open');
      setLastError(null);

      // Send START_SESSION immediately after connect (spec §3.3)
      ws.send(JSON.stringify({
        type:    'START_SESSION',
        payload: { session_id: sessionId },
      }));
    };

    ws.onmessage = (event) => {
      if (unmountedRef.current) return;
      let msg;
      try { msg = JSON.parse(event.data); }
      catch { return; }

      switch (msg.type) {
        case 'FEEDBACK':
          // Persist last question so we can restore it on reconnect
          if (msg.payload?.next_question) {
            sessionStorage.setItem(
              `session:${sessionId}:lastQuestion`,
              JSON.stringify(msg.payload.next_question),
            );
          }
          onFeedbackRef.current?.(msg.payload);
          break;

        case 'SESSION_ENDED':
          sessionStorage.removeItem(`session:${sessionId}:lastQuestion`);
          setStatus('closed');
          onSessionEndedRef.current?.(msg.payload);
          ws.close();
          break;

        case 'ERROR':
          setLastError(msg.payload);
          onErrorRef.current?.(msg.payload);
          break;

        default:
          break;
      }
    };

    ws.onclose = (event) => {
      if (unmountedRef.current) return;
      // 1000 = normal close (session ended intentionally)
      if (event.code === 1000) { setStatus('closed'); return; }
      // 401 = JWT rejected by server — don't retry
      if (event.code === 4001 || event.reason === 'Unauthorized') {
        setStatus('error');
        setLastError({ code: 'auth_error', message: 'Session authentication failed' });
        return;
      }
      _scheduleReconnect();
    };

    ws.onerror = () => {
      // onerror always fires before onclose — let onclose handle retry logic
      if (unmountedRef.current) return;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, token, enabled]);

  const _scheduleReconnect = useCallback(() => {
    if (retriesRef.current >= MAX_RETRIES) {
      setStatus('error');
      setLastError({ code: 'max_retries', message: 'Connection lost. Please refresh.' });
      return;
    }
    retriesRef.current += 1;
    const delay = BASE_DELAY_MS * Math.pow(2, retriesRef.current - 1);
    setStatus('reconnecting');
    timerRef.current = setTimeout(() => {
      if (!unmountedRef.current) connect();
    }, delay);
  }, [connect]);

  // Initial connection
  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      clearTimeout(timerRef.current);
      wsRef.current?.close(1000, 'component unmounted');
    };
  }, [connect]);

  /**
   * Send a JSON frame to the server.
   * Silently drops the message if the socket is not open.
   */
  const send = useCallback((type, payload = {}) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type, payload }));
  }, []);

  /**
   * Restore the last question from sessionStorage (for reconnect UI).
   */
  const getRestoredQuestion = useCallback(() => {
    try {
      const raw = sessionStorage.getItem(`session:${sessionId}:lastQuestion`);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }, [sessionId]);

  return { send, status, lastError, getRestoredQuestion };
}
