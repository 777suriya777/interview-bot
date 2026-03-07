/**
 * FeedbackCard.jsx
 *
 * Displays the AI evaluation of a single answer:
 *   - Four dimension score bars (content / relevance / completeness / accuracy)
 *   - Overall score ring
 *   - Confidence badge (high / moderate / low / anxious)
 *   - Hedging words highlighted inline
 *   - Delivery flags (voice only: fast_speech, nervous_pitch, etc.)
 *   - Improvement tips list
 *
 * Props:
 *   feedback — the FEEDBACK payload from the WebSocket server
 *   inputMode — 'text' | 'voice'
 */

import { useEffect, useRef } from 'react';

// ── Score dimension metadata ─────────────────────────────────────────────────
const DIMS = [
  { key: 'content',      label: 'Content',      color: 'bg-violet-500' },
  { key: 'relevance',    label: 'Relevance',    color: 'bg-sky-500'    },
  { key: 'completeness', label: 'Completeness', color: 'bg-emerald-500'},
  { key: 'accuracy',     label: 'Accuracy',     color: 'bg-amber-500'  },
];

// ── Confidence badge config ───────────────────────────────────────────────────
const CONFIDENCE_CONFIG = {
  high:     { label: 'High Confidence',     cls: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30', dot: 'bg-emerald-400' },
  moderate: { label: 'Moderate Confidence', cls: 'bg-sky-500/15 text-sky-300 ring-sky-500/30',             dot: 'bg-sky-400'     },
  low:      { label: 'Low Confidence',      cls: 'bg-amber-500/15 text-amber-300 ring-amber-500/30',       dot: 'bg-amber-400'   },
  anxious:  { label: 'Anxious Delivery',    cls: 'bg-red-500/15 text-red-300 ring-red-500/30',             dot: 'bg-red-400'     },
};

// ── Delivery flag labels ──────────────────────────────────────────────────────
const FLAG_LABELS = {
  fast_speech:       { icon: '⚡', text: 'Speaking too fast (>165 WPM)' },
  nervous_pitch:     { icon: '〰️', text: 'High pitch variation detected' },
  excessive_pauses:  { icon: '⏸',  text: 'Too many pauses' },
  low_volume:        { icon: '🔇', text: 'Very flat voice level' },
  too_short:         { icon: '📏', text: 'Answer too brief (<30 words)' },
};

// ── ScoreBar ─────────────────────────────────────────────────────────────────
function ScoreBar({ label, score, color, index }) {
  const barRef = useRef(null);

  useEffect(() => {
    if (!barRef.current) return;
    // Animate width in via requestAnimationFrame after mount
    const pct = (score / 4) * 100;
    barRef.current.style.width = '0%';
    const id = setTimeout(() => {
      if (barRef.current) barRef.current.style.width = `${pct}%`;
    }, 60 + index * 80);
    return () => clearTimeout(id);
  }, [score, index]);

  const pct = Math.round((score / 4) * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs font-medium tracking-wide text-slate-400 uppercase">{label}</span>
        <span className="text-sm font-bold text-slate-200 tabular-nums">{score}<span className="text-slate-500 font-normal">/4</span></span>
      </div>
      <div className="h-1.5 bg-slate-700/60 rounded-full overflow-hidden">
        <div
          ref={barRef}
          className={`h-full rounded-full transition-[width] duration-700 ease-out ${color}`}
          style={{ width: 0 }}
          role="progressbar"
          aria-valuenow={score}
          aria-valuemin={0}
          aria-valuemax={4}
          aria-label={`${label} score ${score} out of 4`}
        />
      </div>
      <div className="text-right">
        <span className="text-[10px] text-slate-600">{pct}%</span>
      </div>
    </div>
  );
}

// ── OverallRing ───────────────────────────────────────────────────────────────
function OverallRing({ score }) {
  const pct     = (score / 4) * 100;
  const radius  = 28;
  const circ    = 2 * Math.PI * radius;
  const offset  = circ - (pct / 100) * circ;

  const color =
    score >= 3.5 ? '#34d399' :
    score >= 2.5 ? '#38bdf8' :
    score >= 1.5 ? '#fbbf24' : '#f87171';

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="72" height="72" className="-rotate-90">
        <circle cx="36" cy="36" r={radius} fill="none" stroke="#1e293b" strokeWidth="6" />
        <circle
          cx="36" cy="36" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.9s ease-out' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center" style={{ marginTop: -4 }}>
        {/* positioned via parent relative */}
      </div>
      <span className="text-lg font-bold text-slate-100 -mt-1">{score.toFixed(1)}</span>
      <span className="text-[10px] text-slate-500 uppercase tracking-widest">Overall</span>
    </div>
  );
}

// ── Main FeedbackCard ─────────────────────────────────────────────────────────
export default function FeedbackCard({ feedback, inputMode = 'text' }) {
  if (!feedback) return null;

  const { scores, confidence_label, hedging_words = [], delivery_flags = [], improvement_tips = [], _warning } = feedback;
  const confCfg = CONFIDENCE_CONFIG[confidence_label] || CONFIDENCE_CONFIG.moderate;

  return (
    <div className="rounded-2xl bg-slate-800/70 border border-slate-700/50 shadow-xl shadow-black/30 overflow-hidden">

      {/* ── Header bar ───────────────────────────────────────────────── */}
      <div className="px-5 py-3 bg-slate-900/60 border-b border-slate-700/40 flex items-center justify-between">
        <span className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Answer Evaluation</span>
        {_warning && (
          <span className="text-[10px] text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded-full border border-amber-400/20">
            ⚠ Partial evaluation
          </span>
        )}
      </div>

      <div className="p-5 grid grid-cols-[1fr_auto] gap-6">

        {/* ── Score bars ───────────────────────────────────────────────── */}
        <div className="space-y-4">
          {DIMS.map((dim, i) => (
            <ScoreBar
              key={dim.key}
              label={dim.label}
              score={scores[dim.key] ?? 0}
              color={dim.color}
              index={i}
            />
          ))}
        </div>

        {/* ── Overall ring ─────────────────────────────────────────────── */}
        <div className="flex items-center">
          <OverallRing score={scores.overall ?? 0} />
        </div>
      </div>

      {/* ── Confidence badge ─────────────────────────────────────────── */}
      <div className="px-5 pb-4 flex flex-wrap gap-2 items-center">
        <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ring-1 ${confCfg.cls}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${confCfg.dot}`} />
          {confCfg.label}
        </span>

        {/* Delivery flags (voice only) */}
        {inputMode === 'voice' && delivery_flags.map((flag) => {
          const cfg = FLAG_LABELS[flag] || { icon: '⚠', text: flag };
          return (
            <span key={flag} className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-red-500/10 text-red-300 ring-1 ring-red-500/20">
              <span>{cfg.icon}</span> {cfg.text}
            </span>
          );
        })}
      </div>

      {/* ── Hedging words ─────────────────────────────────────────────── */}
      {hedging_words.length > 0 && (
        <div className="px-5 pb-4">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Hedging language detected</p>
          <div className="flex flex-wrap gap-1.5">
            {hedging_words.map((hw, i) => (
              <span key={i} className="text-xs px-2 py-0.5 rounded bg-amber-400/10 text-amber-300 border border-amber-400/20 font-mono">
                "{hw.word || hw.text}"
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Improvement tips ──────────────────────────────────────────── */}
      {improvement_tips.length > 0 && (
        <div className="px-5 pb-5 border-t border-slate-700/40 pt-4">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-3">Suggestions</p>
          <ul className="space-y-2">
            {improvement_tips.map((tip, i) => (
              <li key={i} className="flex gap-2.5 text-sm text-slate-300 leading-relaxed">
                <span className="mt-0.5 flex-shrink-0 w-4 h-4 rounded-full bg-violet-500/20 text-violet-400 flex items-center justify-center text-[10px] font-bold">
                  {i + 1}
                </span>
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
