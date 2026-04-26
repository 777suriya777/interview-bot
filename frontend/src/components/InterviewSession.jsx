/**
 * InterviewSession.jsx
 *
 * The main interview screen. Manages:
 *   - WebSocket connection via useWebSocket hook
 *   - Current question display with difficulty badge and type chip
 *   - Text answer input (textarea + submit)
 *   - Voice recording (MediaRecorder → base64 webm → SUBMIT_VOICE)
 *   - Per-answer FeedbackCard with slide-in animation
 *   - Session progress indicator
 *   - Reconnecting / error / session-ended states
 *
 * Props:
 *   sessionId    — UUID string
 *   firstQuestion — {id, text, type, difficulty} from POST /api/sessions response
 *   token        — JWT
 *   onSessionEnd — callback(reportUrl) when session completes
 *   onExit       — callback() for manual early exit
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import FeedbackCard from './FeedbackCard';
import { useWebSocket } from '../hooks/useWebSocket';

// ── Difficulty badge config ───────────────────────────────────────────────────
const DIFF_CONFIG = {
  1: { label: 'Beginner',     cls: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30' },
  2: { label: 'Easy',         cls: 'bg-sky-500/15 text-sky-300 ring-sky-500/30'             },
  3: { label: 'Intermediate', cls: 'bg-violet-500/15 text-violet-300 ring-violet-500/30'    },
  4: { label: 'Hard',         cls: 'bg-amber-500/15 text-amber-300 ring-amber-500/30'       },
  5: { label: 'Expert',       cls: 'bg-red-500/15 text-red-300 ring-red-500/30'             },
};

const TYPE_LABELS = {
  technical:   'Technical',
  behavioural: 'Behavioural',
  hr:          'HR',
};

// ── Voice recording hook ──────────────────────────────────────────────────────
function useVoiceRecorder() {
  const [recording,      setRecording]      = useState(false);
  const [audioSupported, setAudioSupported] = useState(true);
  const [recordingMs,    setRecordingMs]    = useState(0);
  const recorderRef  = useRef(null);
  const chunksRef    = useRef([]);
  const timerRef     = useRef(null);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) setAudioSupported(false);
    return () => {
      clearInterval(timerRef.current);
      recorderRef.current?.stream?.getTracks().forEach(t => t.stop());
    };
  }, []);

  const start = useCallback(async () => {
    if (!audioSupported || recording) return;
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current  = [];
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.start(250);  // collect chunks every 250ms
      setRecording(true);
      setRecordingMs(0);
      timerRef.current = setInterval(() => setRecordingMs(ms => ms + 100), 100);
    } catch (err) {
      console.error('Microphone access denied:', err);
      setAudioSupported(false);
    }
  }, [audioSupported, recording]);

  /**
   * Stop recording and return { audioB64, durationSeconds } or null on error.
   */
  const stop = useCallback(() => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state === 'inactive') { resolve(null); return; }

      clearInterval(timerRef.current);
      const duration = recordingMs / 1000;

      recorder.onstop = () => {
        const blob   = new Blob(chunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
          // Strip the data URL prefix (data:audio/webm;base64,)
          const base64 = reader.result.split(',')[1];
          resolve({ audioB64: base64, durationSeconds: duration });
        };
        reader.readAsDataURL(blob);
        // Stop all microphone tracks
        recorder.stream.getTracks().forEach(t => t.stop());
      };
      recorder.stop();
      setRecording(false);
    });
  }, [recordingMs]);

  return { start, stop, recording, audioSupported, recordingMs };
}

// ── RecordingTimer ────────────────────────────────────────────────────────────
function RecordingTimer({ ms }) {
  const s   = Math.floor(ms / 1000);
  const min = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = (s % 60).toString().padStart(2, '0');
  return (
    <span className="font-mono text-sm text-red-400 tabular-nums tracking-wider">
      {min}:{sec}
    </span>
  );
}

// ── ConnectionBanner ──────────────────────────────────────────────────────────
function ConnectionBanner({ status, lastError }) {
  if (status === 'open') return null;

  const cfg = {
    connecting:   { bg: 'bg-slate-700/80',   text: 'Connecting…',          icon: '○' },
    reconnecting: { bg: 'bg-amber-900/60',   text: 'Reconnecting…',        icon: '↻' },
    error:        { bg: 'bg-red-900/60',     text: lastError?.message || 'Connection error', icon: '✗' },
    closed:       { bg: 'bg-slate-800/80',   text: 'Session closed',       icon: '■' },
  }[status] || { bg: 'bg-slate-700/80', text: status, icon: '…' };

  return (
    <div className={`${cfg.bg} border-b border-white/5 px-4 py-2 flex items-center gap-2 text-sm text-slate-300`}>
      <span className={status === 'reconnecting' ? 'animate-spin inline-block' : ''}>{cfg.icon}</span>
      {cfg.text}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function InterviewSession({
  sessionId,
  firstQuestion,
  token,
  onSessionEnd,
  onExit,
}) {
  const [currentQuestion, setCurrentQuestion] = useState(firstQuestion);
  const [feedback,        setFeedback]        = useState(null);
  const [inputMode,       setInputMode]       = useState('text');
  const [answerText,      setAnswerText]      = useState('');
  const [submitting,      setSubmitting]      = useState(false);
  const [questionCount,   setQuestionCount]   = useState(1);
  const [sessionEnded,    setSessionEnded]    = useState(false);
  const [reportUrl,       setReportUrl]       = useState(null);
  const [ttsEnabled,      setTtsEnabled]      = useState(true);
  const textareaRef = useRef(null);

  const voice = useVoiceRecorder();

  // ── Text-to-Speech (TTS) ───────────────────────────────────────────────────
  useEffect(() => {
    if (ttsEnabled && currentQuestion?.text) {
      // Cancel any ongoing speech so they don't overlap
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(currentQuestion.text);
      window.speechSynthesis.speak(utterance);
    }
  }, [currentQuestion, ttsEnabled]);

  // Clean up TTS when unmounting
  useEffect(() => {
    return () => window.speechSynthesis.cancel();
  }, []);

  // ── Feedback from server ──────────────────────────────────────────────────
  const handleFeedback = useCallback((payload) => {
    setFeedback(payload);
    setSubmitting(false);
    setAnswerText('');
    if (payload.next_question) {
      setCurrentQuestion(payload.next_question);
      setQuestionCount(n => n + 1);
    }
    // Scroll to feedback after short delay for animation
    setTimeout(() => {
      document.getElementById('feedback-anchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
  }, []);

  const handleSessionEnded = useCallback((payload) => {
    setSessionEnded(true);
    setReportUrl(payload.report_url);
    onSessionEnd?.(payload.report_url);
  }, [onSessionEnd]);

  const handleError = useCallback((err) => {
    setSubmitting(false);
    // asr_timeout: clear voice state so user can retry
    if (err.code === 'asr_timeout') {
      setInputMode('text');
    }
  }, []);

  const { send, status, lastError, getRestoredQuestion } = useWebSocket({
    sessionId,
    token,
    onFeedback:     handleFeedback,
    onSessionEnded: handleSessionEnded,
    onError:        handleError,
  });

  // Restore last question on reconnect
  useEffect(() => {
    if (status === 'open') {
      const restored = getRestoredQuestion();
      if (restored && restored.id !== currentQuestion?.id) {
        setCurrentQuestion(restored);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Auto-focus textarea when switching to text mode
  useEffect(() => {
    if (inputMode === 'text') textareaRef.current?.focus();
  }, [inputMode]);

  // ── Submit handlers ───────────────────────────────────────────────────────
  const submitText = useCallback(() => {
    const trimmed = answerText.trim();
    if (!trimmed || submitting || !currentQuestion) return;
    setSubmitting(true);
    setFeedback(null);
    send('SUBMIT_TEXT', {
      question_id: currentQuestion.id,
      answer_text: trimmed,
    });
  }, [answerText, submitting, currentQuestion, send]);

  const startVoice = useCallback(async () => {
    if (submitting) return;
    setFeedback(null);
    await voice.start();
  }, [submitting, voice]);

  const stopAndSubmitVoice = useCallback(async () => {
    const result = await voice.stop();
    if (!result || !currentQuestion) return;
    setSubmitting(true);
    send('SUBMIT_VOICE', {
      question_id:      currentQuestion.id,
      audio_b64:        result.audioB64,
      duration_seconds: result.durationSeconds,
    });
  }, [voice, currentQuestion, send]);

  const endSession = useCallback(() => {
    send('END_SESSION', {});
  }, [send]);

  const flagQuestion = useCallback(() => {
    if (!currentQuestion) return;
    send('FLAG_QUESTION', { question_id: currentQuestion.id, reason: 'irrelevant' });
  }, [currentQuestion, send]);

  // ── Keyboard shortcut: Ctrl+Enter submits ─────────────────────────────────
  const onKeyDown = useCallback((e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') submitText();
  }, [submitText]);

  // ── Session ended screen ───────────────────────────────────────────────────
  if (sessionEnded) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="w-16 h-16 rounded-full bg-emerald-500/15 ring-1 ring-emerald-500/30 flex items-center justify-center mx-auto text-3xl">
            ✓
          </div>
          <h2 className="text-2xl font-bold text-slate-100">Session Complete</h2>
          <p className="text-slate-400">Your performance report is ready. You answered {questionCount - 1} question{questionCount !== 2 ? 's' : ''}.</p>
          <div className="flex flex-col gap-3">
            {reportUrl && (
              <a
                href={reportUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold transition-colors"
              >
                Download Report PDF
              </a>
            )}
            <button
              onClick={onExit}
              className="w-full py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  const diffCfg = DIFF_CONFIG[currentQuestion?.difficulty] || DIFF_CONFIG[3];

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">

      {/* ── Connection status banner ─────────────────────────────────── */}
      <ConnectionBanner status={status} lastError={lastError} />

      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-4 border-b border-slate-800/60 bg-slate-900/40">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
            Question {questionCount}
          </span>
          {currentQuestion && (
            <>
              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ring-1 ${diffCfg.cls}`}>
                {diffCfg.label}
              </span>
              <span className="text-[11px] text-slate-600 px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700">
                {TYPE_LABELS[currentQuestion.type] || currentQuestion.type}
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (ttsEnabled) window.speechSynthesis.cancel();
              else if (currentQuestion) window.speechSynthesis.speak(new SpeechSynthesisUtterance(currentQuestion.text));
              setTtsEnabled(!ttsEnabled);
            }}
            title={ttsEnabled ? "Disable Read Aloud" : "Enable Read Aloud"}
            className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-all
              ${ttsEnabled 
                ? 'bg-violet-500/15 text-violet-300 border-violet-500/30' 
                : 'bg-transparent text-slate-500 border-transparent hover:bg-slate-800'}`}
          >
            {ttsEnabled ? '🔊 Sound On' : '🔇 Sound Off'}
          </button>
          
          {currentQuestion && (
            <button
              onClick={flagQuestion}
              title="Flag this question as low quality"
              className="text-slate-600 hover:text-amber-400 transition-colors text-sm px-2 py-1 rounded hover:bg-slate-800"
            >
              ⚑ Flag
            </button>
          )}
          <button
            onClick={endSession}
            className="text-xs text-slate-500 hover:text-red-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-red-500/10 border border-transparent hover:border-red-500/20"
          >
            End Session
          </button>
        </div>
      </header>

      {/* ── Main content ─────────────────────────────────────────────── */}
      <main className="flex-1 max-w-3xl w-full mx-auto px-5 py-8 space-y-6">

        {/* Question card */}
        {currentQuestion ? (
          <div className="rounded-2xl bg-slate-900/60 border border-slate-700/40 p-6 shadow-lg">
            <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-3">Your Question</p>
            <p className="text-lg text-slate-100 leading-relaxed font-medium">
              {currentQuestion.text}
            </p>
          </div>
        ) : (
          <div className="rounded-2xl bg-slate-900/60 border border-slate-700/40 p-6 animate-pulse">
            <div className="h-5 bg-slate-800 rounded w-3/4" />
          </div>
        )}

        {/* Input mode toggle */}
        <div className="flex gap-1 p-1 rounded-xl bg-slate-900 border border-slate-800 w-fit">
          {['text', 'voice'].map(mode => (
            <button
              key={mode}
              onClick={() => setInputMode(mode)}
              disabled={voice.recording || submitting}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all capitalize
                ${inputMode === mode
                  ? 'bg-slate-700 text-slate-100 shadow-sm'
                  : 'text-slate-500 hover:text-slate-300'
                }
                disabled:opacity-40`}
            >
              {mode === 'voice' && !voice.audioSupported ? 'Voice (unavailable)' : mode}
            </button>
          ))}
        </div>

        {/* Text input */}
        {inputMode === 'text' && (
          <div className="space-y-3">
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={answerText}
                onChange={e => setAnswerText(e.target.value)}
                onKeyDown={onKeyDown}
                disabled={submitting || status !== 'open'}
                placeholder="Type your answer here… (Ctrl+Enter to submit)"
                rows={6}
                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder-slate-600 text-sm leading-relaxed resize-none focus:outline-none focus:ring-1 focus:ring-violet-500/60 focus:border-violet-500/40 transition-all disabled:opacity-50"
              />
              <span className={`absolute bottom-3 right-4 text-[10px] tabular-nums transition-colors
                ${answerText.split(/\s+/).filter(Boolean).length < 30 ? 'text-amber-500' : 'text-slate-600'}`}>
                {answerText.split(/\s+/).filter(Boolean).length} words
              </span>
            </div>
            <button
              onClick={submitText}
              disabled={!answerText.trim() || submitting || status !== 'open'}
              className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-600
                text-white font-semibold text-sm transition-all active:scale-[0.98] disabled:cursor-not-allowed shadow-lg shadow-violet-900/30"
            >
              {submitting ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Evaluating…
                </span>
              ) : 'Submit Answer'}
            </button>
          </div>
        )}

        {/* Voice input */}
        {inputMode === 'voice' && voice.audioSupported && (
          <div className="flex flex-col items-center gap-5 py-4">
            {voice.recording ? (
              <>
                {/* Pulsing record indicator */}
                <div className="relative">
                  <div className="w-20 h-20 rounded-full bg-red-500/20 animate-ping absolute inset-0" />
                  <div className="relative w-20 h-20 rounded-full bg-red-500/30 border-2 border-red-500/60 flex items-center justify-center">
                    <div className="w-5 h-5 rounded-sm bg-red-400" />
                  </div>
                </div>
                <RecordingTimer ms={voice.recordingMs} />
                <button
                  onClick={stopAndSubmitVoice}
                  disabled={submitting}
                  className="px-8 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold text-sm transition-all active:scale-[0.98] shadow-lg shadow-red-900/30"
                >
                  Stop & Submit
                </button>
                <p className="text-xs text-slate-500">
                  {voice.recordingMs < 2000
                    ? 'Keep speaking — minimum 2 seconds'
                    : 'Recording… click Stop when finished'}
                </p>
              </>
            ) : (
              <>
                <button
                  onClick={startVoice}
                  disabled={submitting || status !== 'open'}
                  className="w-20 h-20 rounded-full bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800
                    flex items-center justify-center text-3xl transition-all active:scale-95
                    shadow-lg shadow-violet-900/40 disabled:shadow-none disabled:text-slate-600"
                >
                  🎙
                </button>
                <p className="text-sm text-slate-400">
                  {submitting ? 'Evaluating your answer…' : 'Tap to start recording'}
                </p>
                {submitting && (
                  <div className="w-5 h-5 border-2 border-violet-500/30 border-t-violet-400 rounded-full animate-spin" />
                )}
              </>
            )}
          </div>
        )}

        {/* Feedback card */}
        {feedback && (
          <div id="feedback-anchor" className="animate-[slideUp_0.35s_ease-out]">
            <FeedbackCard feedback={feedback} inputMode={inputMode} />
          </div>
        )}
      </main>

      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
