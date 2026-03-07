/**
 * Auth.jsx
 *
 * Combined login/register page. Aesthetic direction: dark editorial —
 * charcoal backgrounds, sharp typographic hierarchy, a single violet accent,
 * subtle grain texture overlay. Functional and calm — no flashy animations.
 *
 * Props:
 *   onAuth(token, userId) — called after successful login or register
 */

import { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';

// ── Field component ───────────────────────────────────────────────────────────
function Field({ label, id, type = 'text', value, onChange, error, autoComplete, placeholder }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-semibold uppercase tracking-widest text-slate-400">
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        autoComplete={autoComplete}
        placeholder={placeholder}
        className={`w-full bg-slate-900 border rounded-xl px-4 py-3 text-slate-100 text-sm
          placeholder-slate-600 focus:outline-none focus:ring-1 transition-all
          ${error
            ? 'border-red-500/50 focus:ring-red-500/40 focus:border-red-500/50'
            : 'border-slate-700 focus:ring-violet-500/50 focus:border-violet-500/40'
          }`}
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}

// ── Login form ────────────────────────────────────────────────────────────────
function LoginForm({ onAuth, onSwitch }) {
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [errors,   setErrors]   = useState({});
  const [loading,  setLoading]  = useState(false);
  const [apiError, setApiError] = useState('');

  const validate = () => {
    const e = {};
    if (!email.trim())         e.email    = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(email)) e.email = 'Enter a valid email';
    if (!password)             e.password = 'Password is required';
    return e;
  };

  const submit = async (ev) => {
    ev.preventDefault();
    const e = validate();
    if (Object.keys(e).length) { setErrors(e); return; }
    setErrors({});
    setApiError('');
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/api/auth/login`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setApiError(data.detail || 'Invalid email or password');
        return;
      }
      onAuth(data.access_token, data.user_id);
    } catch {
      setApiError('Unable to connect. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-5" noValidate>
      <Field
        label="Email" id="login-email" type="email"
        value={email} onChange={setEmail}
        error={errors.email}
        autoComplete="email" placeholder="you@example.com"
      />
      <Field
        label="Password" id="login-password" type="password"
        value={password} onChange={setPassword}
        error={errors.password}
        autoComplete="current-password" placeholder="••••••••"
      />
      {apiError && (
        <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {apiError}
        </div>
      )}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800
          disabled:text-slate-500 text-white font-bold text-sm transition-all
          active:scale-[0.98] shadow-lg shadow-violet-900/30 mt-2"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            Signing in…
          </span>
        ) : 'Sign In'}
      </button>
      <p className="text-center text-sm text-slate-500">
        No account?{' '}
        <button type="button" onClick={onSwitch}
          className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
          Create one
        </button>
      </p>
    </form>
  );
}

// ── Register form ─────────────────────────────────────────────────────────────
function RegisterForm({ onAuth, onSwitch }) {
  const [name,     setName]     = useState('');
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [errors,   setErrors]   = useState({});
  const [loading,  setLoading]  = useState(false);
  const [apiError, setApiError] = useState('');

  const validate = () => {
    const e = {};
    if (!name.trim())          e.name     = 'Name is required';
    if (!email.trim())         e.email    = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(email)) e.email = 'Enter a valid email';
    if (!password)             e.password = 'Password is required';
    else if (password.length < 8)         e.password = 'Minimum 8 characters';
    return e;
  };

  const submit = async (ev) => {
    ev.preventDefault();
    const e = validate();
    if (Object.keys(e).length) { setErrors(e); return; }
    setErrors({});
    setApiError('');
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/api/auth/register`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ name: name.trim(), email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setApiError(
          res.status === 409
            ? 'An account with this email already exists.'
            : (data.detail || 'Registration failed. Please try again.')
        );
        return;
      }
      onAuth(data.access_token, data.user_id);
    } catch {
      setApiError('Unable to connect. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-5" noValidate>
      <Field
        label="Full Name" id="reg-name"
        value={name} onChange={setName}
        error={errors.name}
        autoComplete="name" placeholder="Ada Lovelace"
      />
      <Field
        label="Email" id="reg-email" type="email"
        value={email} onChange={setEmail}
        error={errors.email}
        autoComplete="email" placeholder="you@example.com"
      />
      <Field
        label="Password" id="reg-password" type="password"
        value={password} onChange={setPassword}
        error={errors.password}
        autoComplete="new-password" placeholder="Min. 8 characters"
      />
      {apiError && (
        <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {apiError}
        </div>
      )}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800
          disabled:text-slate-500 text-white font-bold text-sm transition-all
          active:scale-[0.98] shadow-lg shadow-violet-900/30 mt-2"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            Creating account…
          </span>
        ) : 'Create Account'}
      </button>
      <p className="text-center text-sm text-slate-500">
        Already have an account?{' '}
        <button type="button" onClick={onSwitch}
          className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
          Sign in
        </button>
      </p>
    </form>
  );
}

// ── Auth page ─────────────────────────────────────────────────────────────────
export default function Auth({ onAuth }) {
  const [mode, setMode] = useState('login');  // 'login' | 'register'

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 relative overflow-hidden">

      {/* Grain texture overlay */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.025]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          backgroundSize: '200px',
        }}
      />

      {/* Ambient glow */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-violet-600/8 rounded-full blur-3xl" />

      {/* Card */}
      <div className="relative w-full max-w-md">

        {/* Wordmark */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-violet-900/50">
              IB
            </div>
            <span className="text-slate-300 font-semibold tracking-tight">InterviewBot</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
            {mode === 'login' ? 'Welcome back' : 'Get started'}
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            {mode === 'login'
              ? 'Sign in to continue your preparation'
              : 'Create an account to start practising'}
          </p>
        </div>

        {/* Form card */}
        <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl p-7 shadow-2xl shadow-black/40 backdrop-blur-sm">
          {mode === 'login'
            ? <LoginForm    onAuth={onAuth} onSwitch={() => setMode('register')} />
            : <RegisterForm onAuth={onAuth} onSwitch={() => setMode('login')} />
          }
        </div>

        {/* Footer note */}
        <p className="text-center text-xs text-slate-700 mt-6">
          AI-NLP Interview Preparation · M.Tech CSE Project
        </p>
      </div>
    </div>
  );
}
