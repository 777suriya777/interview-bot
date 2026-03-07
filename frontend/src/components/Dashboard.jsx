/**
 * Dashboard.jsx
 *
 * Post-login landing screen showing:
 *   - Overall performance score + trend arrow
 *   - Radar chart: per-topic scores (Recharts RadarChart)
 *   - Line chart: overall score across last N sessions (Recharts LineChart)
 *   - Confidence distribution bar chart (Recharts BarChart)
 *   - Priority improvement areas
 *   - Session history table (last 10)
 *   - "New Session" CTA
 *   - Empty state for first-time users
 *
 * Props:
 *   token      — JWT string
 *   userId     — UUID string
 *   onNewSession() — navigate to setup screen
 *   onLogout()     — clear auth and return to login
 */

import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';

// ── Colour palette ────────────────────────────────────────────────────────────
const VIOLET  = '#7c3aed';
const SKY     = '#0ea5e9';
const EMERALD = '#10b981';
const AMBER   = '#f59e0b';
const RED_SFT = '#f87171';

const CONFIDENCE_COLORS = {
  high:     EMERALD,
  moderate: SKY,
  low:      AMBER,
  anxious:  RED_SFT,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function authHeaders(token) {
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: '2-digit' });
}

function fmtDuration(start, end) {
  if (!start || !end) return '—';
  const secs = Math.round((new Date(end) - new Date(start)) / 1000);
  const m    = Math.floor(secs / 60);
  const s    = secs % 60;
  return `${m}m ${s}s`;
}

function scoreColor(score, max = 100) {
  const pct = (score / max) * 100;
  if (pct >= 75) return 'text-emerald-400';
  if (pct >= 50) return 'text-sky-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

// ── Custom tooltip for charts ─────────────────────────────────────────────────
function ChartTooltip({ active, payload, label, unit = '' }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 shadow-xl text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || p.fill }} className="font-semibold">
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}{unit}
        </p>
      ))}
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, accent = false }) {
  return (
    <div className={`rounded-xl p-4 border ${accent
      ? 'bg-violet-600/10 border-violet-500/30'
      : 'bg-slate-900 border-slate-800'}`}>
      <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${accent ? 'text-violet-300' : 'text-slate-100'}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

// ── Section heading ───────────────────────────────────────────────────────────
function SectionHeading({ children }) {
  return (
    <h2 className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 mb-4">
      {children}
    </h2>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ onNewSession }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-violet-600/10 border border-violet-500/20 flex items-center justify-center text-3xl">
        🎙
      </div>
      <div>
        <h2 className="text-xl font-bold text-slate-200 mb-1">No sessions yet</h2>
        <p className="text-slate-500 text-sm max-w-xs">
          Start your first practice interview to see your performance data here.
        </p>
      </div>
      <button
        onClick={onNewSession}
        className="px-6 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold text-sm
          transition-all active:scale-[0.98] shadow-lg shadow-violet-900/30"
      >
        Begin First Session →
      </button>
    </div>
  );
}

// ── Priority area chip ────────────────────────────────────────────────────────
function PriorityChip({ topic }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full
      bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
      ↗ {topic.replace(/_/g, ' ')}
    </span>
  );
}

// ── Session row ───────────────────────────────────────────────────────────────
function SessionRow({ session }) {
  const statusColors = {
    completed:  'text-emerald-400',
    active:     'text-sky-400',
    abandoned:  'text-slate-500',
  };

  return (
    <tr className="border-t border-slate-800/60 hover:bg-slate-900/40 transition-colors">
      <td className="py-3 px-4 text-sm text-slate-300 capitalize">
        {session.interview_type}
      </td>
      <td className="py-3 px-4 text-sm text-slate-400">
        {session.target_role || <span className="text-slate-600">—</span>}
      </td>
      <td className="py-3 px-4 text-sm">
        <span className={`font-semibold ${scoreColor(parseFloat(session.performance_score || 0) * 25)}`}>
          {parseFloat(session.performance_score || 0).toFixed(1)}
          <span className="text-slate-600 font-normal">/4</span>
        </span>
      </td>
      <td className="py-3 px-4 text-sm text-slate-400">
        {fmtDuration(session.started_at, session.ended_at)}
      </td>
      <td className="py-3 px-4 text-sm text-slate-500">
        {fmtDate(session.started_at)}
      </td>
      <td className="py-3 px-4">
        <span className={`text-xs font-medium capitalize ${statusColors[session.status] || 'text-slate-500'}`}>
          {session.status}
        </span>
      </td>
      <td className="py-3 px-4">
        {session.report_url ? (
          <a
            href={session.report_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
          >
            PDF ↗
          </a>
        ) : (
          <span className="text-slate-700 text-xs">—</span>
        )}
      </td>
    </tr>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard({ token, userId, onNewSession, onLogout }) {
  const [sessions,     setSessions]     = useState([]);
  const [performance,  setPerformance]  = useState(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState('');

  useEffect(() => {
    let cancelled = false;

    async function fetchAll() {
      setLoading(true);
      setError('');
      try {
        const [sessRes, perfRes] = await Promise.all([
          fetch(`${API_BASE}/api/sessions`,    { headers: authHeaders(token) }),
          fetch(`${API_BASE}/api/performance`, { headers: authHeaders(token) }),
        ]);

        if (cancelled) return;

        if (!sessRes.ok || !perfRes.ok) {
          setError('Failed to load dashboard data.');
          return;
        }

        const { sessions: sessData } = await sessRes.json();
        const perfData               = await perfRes.json();

        setSessions(sessData  || []);
        setPerformance(perfData);
      } catch {
        if (!cancelled) setError('Unable to connect to server.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchAll();
    return () => { cancelled = true; };
  }, [token]);

  // ── Derived chart data ──────────────────────────────────────────────────────

  // Radar: per-topic scores normalised to 0–100
  const radarData = performance
    ? Object.entries(performance.topics).map(([topic, data]) => ({
        topic: topic.replace(/_/g, ' '),
        score: Math.round(data.score),
        fullMark: 100,
      }))
    : [];

  // Line: score trend across completed sessions (oldest → newest)
  const trendData = sessions
    .filter(s => s.status === 'completed')
    .slice()
    .reverse()
    .slice(-10)  // last 10 completed sessions
    .map((s, i) => ({
      session:   `#${i + 1}`,
      score:     parseFloat((parseFloat(s.performance_score || 0) * 25).toFixed(1)),
    }));

  // Confidence distribution: aggregate across all sessions from report data
  // We use the topic breakdown as a proxy since session rows don't carry conf dist
  const confData = performance
    ? [
        { label: 'High',     value: 0, color: EMERALD },
        { label: 'Moderate', value: 0, color: SKY     },
        { label: 'Low',      value: 0, color: AMBER   },
        { label: 'Anxious',  value: 0, color: RED_SFT },
      ]
    : [];

  // Stats
  const completedCount = sessions.filter(s => s.status === 'completed').length;
  const avgScore       = performance?.overall_score ?? 0;
  const priorityAreas  = performance?.priority_areas ?? [];

  // Trend arrow: compare last 2 completed sessions
  const completedSessions = sessions.filter(s => s.status === 'completed');
  let trendArrow = '';
  if (completedSessions.length >= 2) {
    const latest = parseFloat(completedSessions[0].performance_score || 0);
    const prev   = parseFloat(completedSessions[1].performance_score || 0);
    trendArrow = latest > prev ? '↑' : latest < prev ? '↓' : '→';
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60 bg-slate-900/30">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center text-white font-bold text-xs shadow-md shadow-violet-900/40">
            IB
          </div>
          <span className="text-slate-200 font-semibold text-sm">InterviewBot</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onNewSession}
            className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold
              transition-all active:scale-[0.98] shadow-md shadow-violet-900/30"
          >
            + New Session
          </button>
          <button onClick={onLogout} className="text-xs text-slate-600 hover:text-slate-400 transition-colors px-2 py-1">
            Sign out
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">

        {loading && (
          <div className="flex items-center justify-center py-24">
            <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
          </div>
        )}

        {!loading && error && (
          <div className="text-center py-24 space-y-3">
            <p className="text-slate-400">{error}</p>
            <button onClick={() => window.location.reload()}
              className="text-sm text-violet-400 hover:text-violet-300 transition-colors">
              Retry
            </button>
          </div>
        )}

        {!loading && !error && sessions.length === 0 && (
          <EmptyState onNewSession={onNewSession} />
        )}

        {!loading && !error && sessions.length > 0 && (
          <div className="space-y-10">

            {/* Page title */}
            <div className="flex items-end justify-between">
              <div>
                <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Performance Dashboard</h1>
                <p className="text-slate-500 text-sm mt-0.5">{completedCount} session{completedCount !== 1 ? 's' : ''} completed</p>
              </div>
            </div>

            {/* ── Stat cards ──────────────────────────────────────────────── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <StatCard
                label="Overall Score"
                value={`${avgScore.toFixed(1)}%`}
                sub={trendArrow ? `${trendArrow} vs last session` : undefined}
                accent
              />
              <StatCard label="Sessions" value={completedCount} sub="completed" />
              <StatCard
                label="Topics Practised"
                value={radarData.length}
                sub="unique areas"
              />
              <StatCard
                label="Priority Areas"
                value={priorityAreas.length || '—'}
                sub={priorityAreas.length ? 'need focus' : 'all strong'}
              />
            </div>

            {/* ── Priority areas ───────────────────────────────────────────── */}
            {priorityAreas.length > 0 && (
              <div>
                <SectionHeading>Focus Areas</SectionHeading>
                <div className="flex flex-wrap gap-2">
                  {priorityAreas.map(area => <PriorityChip key={area} topic={area} />)}
                </div>
              </div>
            )}

            {/* ── Charts row ───────────────────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* Radar: per-topic */}
              {radarData.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                  <SectionHeading>Topic Breakdown</SectionHeading>
                  <ResponsiveContainer width="100%" height={260}>
                    <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                      <PolarGrid stroke="#1e293b" />
                      <PolarAngleAxis
                        dataKey="topic"
                        tick={{ fill: '#94a3b8', fontSize: 11, textAnchor: 'middle' }}
                      />
                      <Radar
                        name="Score"
                        dataKey="score"
                        stroke={VIOLET}
                        fill={VIOLET}
                        fillOpacity={0.18}
                        strokeWidth={2}
                      />
                      <Tooltip content={<ChartTooltip unit="%" />} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Line: score trend */}
              {trendData.length >= 2 && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                  <SectionHeading>Score Trend</SectionHeading>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={trendData} margin={{ top: 10, right: 16, bottom: 0, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="session"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip content={<ChartTooltip unit="%" />} />
                      <Line
                        type="monotone"
                        dataKey="score"
                        name="Score"
                        stroke={VIOLET}
                        strokeWidth={2.5}
                        dot={{ fill: VIOLET, r: 4, strokeWidth: 0 }}
                        activeDot={{ r: 6, fill: VIOLET }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* ── Topic score bars ─────────────────────────────────────────── */}
            {radarData.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <SectionHeading>Per-Topic Scores</SectionHeading>
                <ResponsiveContainer width="100%" height={Math.max(180, radarData.length * 44)}>
                  <BarChart
                    data={radarData}
                    layout="vertical"
                    margin={{ top: 0, right: 16, bottom: 0, left: 100 }}
                    barSize={10}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 100]}
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="topic"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      width={95}
                    />
                    <Tooltip content={<ChartTooltip unit="%" />} />
                    <Bar dataKey="score" name="Score" radius={[0, 5, 5, 0]}>
                      {radarData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.score >= 75 ? EMERALD : entry.score >= 50 ? VIOLET : entry.score >= 30 ? AMBER : RED_SFT}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* ── Session history ───────────────────────────────────────────── */}
            <div>
              <SectionHeading>Session History</SectionHeading>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-slate-950/50">
                      {['Type', 'Role', 'Score', 'Duration', 'Date', 'Status', 'Report'].map(h => (
                        <th key={h} className="py-3 px-4 text-[10px] uppercase tracking-widest text-slate-600 font-semibold">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.slice(0, 10).map(s => <SessionRow key={s.id} session={s} />)}
                  </tbody>
                </table>
                {sessions.length > 10 && (
                  <div className="px-4 py-3 border-t border-slate-800 text-xs text-slate-600 text-center">
                    Showing 10 of {sessions.length} sessions
                  </div>
                )}
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
