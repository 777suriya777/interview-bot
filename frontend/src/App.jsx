/**
 * App.jsx
 *
 * Root application component. Manages:
 *   - Auth state (JWT in localStorage)
 *   - Screen routing: 'auth' → 'setup' → 'session' → 'dashboard'
 *   - Session creation (POST /api/sessions)
 *   - Passing session context to InterviewSession
 *
 * No react-router — state-machine routing keeps the bundle lean and
 * avoids client-side URL complexity for this single-page tool.
 */

import { useCallback, useEffect, useState } from 'react';
import Auth             from './components/Auth';
import Dashboard        from './components/Dashboard';
import InterviewSession from './components/InterviewSession';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';

// ── SetupScreen ───────────────────────────────────────────────────────────────
// Let the user pick interview type, target role, and duration before starting.
function SetupScreen({ token, onStart, onLogout }) {
  const [interviewType, setInterviewType] = useState('technical');
  const [targetRole,    setTargetRole]    = useState('');
  const [duration,      setDuration]      = useState(30);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState('');

  const start = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE}/api/sessions`, {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          interview_type:   interviewType,
          target_role:      targetRole.trim() || undefined,
          duration_minutes: duration,
        }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || 'Failed to start session'); return; }
      onStart(data.session_id, data.first_question);
    } catch {
      setError('Unable to connect to server. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const TYPE_OPTIONS = [
    { value: 'technical',   label: 'Technical',   icon: '💻', desc: 'Data structures, algorithms, system design' },
    { value: 'behavioural', label: 'Behavioural', icon: '🤝', desc: 'Teamwork, conflict, leadership stories' },
    { value: 'hr',          label: 'HR',          icon: '📋', desc: 'Background, motivation, culture fit' },
    { value: 'mixed',       label: 'Mixed',       icon: '🎯', desc: 'Combination of all three types' },
  ];

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center text-white font-bold text-xs">IB</div>
          <span className="text-slate-300 font-semibold text-sm">InterviewBot</span>
        </div>
        <button onClick={onLogout} className="text-xs text-slate-600 hover:text-slate-400 transition-colors">
          Sign out
        </button>
      </header>

      <main className="flex-1 max-w-2xl mx-auto w-full px-6 py-12 space-y-10">
        <div>
          <h1 className="text-3xl font-bold text-slate-100 tracking-tight">New Session</h1>
          <p className="text-slate-500 mt-1 text-sm">Configure your practice interview and begin.</p>
        </div>

        {/* Interview type */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500">Interview Type</h2>
          <div className="grid grid-cols-2 gap-3">
            {TYPE_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setInterviewType(opt.value)}
                className={`text-left p-4 rounded-xl border transition-all
                  ${interviewType === opt.value
                    ? 'bg-violet-600/15 border-violet-500/40 ring-1 ring-violet-500/30'
                    : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
              >
                <div className="text-xl mb-2">{opt.icon}</div>
                <div className="font-semibold text-sm text-slate-200">{opt.label}</div>
                <div className="text-xs text-slate-500 mt-0.5 leading-relaxed">{opt.desc}</div>
              </button>
            ))}
          </div>
        </section>

        {/* Target role */}
        <section className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-widest text-slate-500" htmlFor="target-role">
            Target Role <span className="text-slate-700 font-normal">(optional)</span>
          </label>
          <input
            id="target-role"
            value={targetRole}
            onChange={e => setTargetRole(e.target.value)}
            placeholder="e.g. Software Engineer, Data Scientist…"
            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 text-sm
              placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-violet-500/50 focus:border-violet-500/40 transition-all"
          />
        </section>

        {/* Duration */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500">Duration</h2>
          <div className="flex gap-3">
            {[20, 30, 45].map(d => (
              <button
                key={d}
                onClick={() => setDuration(d)}
                className={`flex-1 py-2.5 rounded-xl text-sm font-medium border transition-all
                  ${duration === d
                    ? 'bg-violet-600/15 border-violet-500/40 text-violet-300 ring-1 ring-violet-500/30'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
              >
                {d} min
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        <button
          onClick={start}
          disabled={loading}
          className="w-full py-4 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800
            disabled:text-slate-500 text-white font-bold transition-all active:scale-[0.98]
            shadow-lg shadow-violet-900/30 disabled:shadow-none text-sm"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
              Starting session…
            </span>
          ) : 'Begin Interview →'}
        </button>
      </main>
    </div>
  );
}

// ── SessionEndedScreen ────────────────────────────────────────────────────────
function SessionEndedScreen({ reportUrl, onNewSession, onLogout }) {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 gap-6">
      <div className="w-14 h-14 rounded-full bg-emerald-500/15 ring-1 ring-emerald-500/30 flex items-center justify-center text-2xl">
        ✓
      </div>
      <div className="text-center">
        <h2 className="text-2xl font-bold text-slate-100 mb-1">Session Complete</h2>
        <p className="text-slate-500 text-sm">Your report has been generated.</p>
      </div>
      <div className="flex flex-col gap-3 w-full max-w-xs">
        {reportUrl && (
          <a
            href={reportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold text-sm text-center transition-colors"
          >
            Download PDF Report
          </a>
        )}
        <button
          onClick={onNewSession}
          className="w-full py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-sm transition-colors"
        >
          Start New Session
        </button>
        <button
          onClick={onLogout}
          className="text-xs text-slate-600 hover:text-slate-400 transition-colors pt-1"
        >
          Sign out
        </button>
      </div>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  // 'auth' | 'setup' | 'session' | 'ended'
  const [screen, setScreen] = useState('auth');

  // Auth
  const [token,  setToken]  = useState(() => localStorage.getItem('ib_token')  || '');
  const [userId, setUserId] = useState(() => localStorage.getItem('ib_user_id') || '');

  // Session
  const [sessionId,      setSessionId]      = useState('');
  const [firstQuestion,  setFirstQuestion]  = useState(null);
  const [reportUrl,      setReportUrl]      = useState('');

  // Restore auth from localStorage
  useEffect(() => {
    if (token && userId) setScreen('dashboard');
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleAuth = useCallback((newToken, newUserId) => {
    localStorage.setItem('ib_token',   newToken);
    localStorage.setItem('ib_user_id', newUserId);
    setToken(newToken);
    setUserId(newUserId);
    setScreen('dashboard');
  }, []);

  const handleLogout = useCallback(() => {
    localStorage.removeItem('ib_token');
    localStorage.removeItem('ib_user_id');
    setToken('');
    setUserId('');
    setScreen('auth');
  }, []);

  const handleStart = useCallback((sId, firstQ) => {
    setSessionId(sId);
    setFirstQuestion(firstQ);
    setScreen('session');
  }, []);

  const handleSessionEnd = useCallback((url) => {
    setReportUrl(url || '');
    setScreen('ended');
  }, []);

  if (screen === 'auth') return <Auth onAuth={handleAuth} />;

  if (screen === 'dashboard') return (
    <Dashboard
      token={token}
      userId={userId}
      onNewSession={() => setScreen('setup')}
      onLogout={handleLogout}
    />
  );

  if (screen === 'setup') return (
    <SetupScreen
      token={token}
      onStart={handleStart}
      onLogout={handleLogout}
    />
  );

  if (screen === 'session') return (
    <InterviewSession
      sessionId={sessionId}
      firstQuestion={firstQuestion}
      token={token}
      onSessionEnd={handleSessionEnd}
      onExit={() => setScreen('dashboard')}
    />
  );

  if (screen === 'ended') return (
    <SessionEndedScreen
      reportUrl={reportUrl}
      onNewSession={() => setScreen('dashboard')}
      onLogout={handleLogout}
    />
  );

  return null;
}
