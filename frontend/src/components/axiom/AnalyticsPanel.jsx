import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, Activity, ArrowLeft, MessageSquare, ListChecks } from 'lucide-react';
import { fetchJson } from '../../lib/api';

/**
 * Analytics panel (Wave 5, D1).
 *
 * Composes the three PG-backed read surfaces — GET /api/stats (the exact
 * 7-field contract), GET /api/feedback/summary, GET /api/eval/runs — into
 * one usage view. Component-local state only; CSS-bar visuals, no chart
 * library. Counts degrade to process-local fallbacks when PostgreSQL is
 * down (the /stats contract mirrors the server comment verbatim); runs
 * listed here are the ones executed against this deployment's PG — the
 * nightly workflow's runs are not persisted anywhere visible.
 */

const STAT_CARDS = [
  { key: 'indexed_documents', label: 'Indexed docs' },
  { key: 'bm25_doc_count', label: 'BM25 docs' },
  { key: 'vector_doc_count', label: 'Vector docs' },
  { key: 'cache_entries', label: 'Cache entries' },
  { key: 'cache_hits', label: 'Cache hits' },
  { key: 'total_queries_processed', label: 'Queries processed' },
];

function formatCount(value) {
  return typeof value === 'number' ? String(value) : '—';
}

function formatStartedAt(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
}

function isQuiet(stats) {
  return (
    stats.indexed_documents === 0 &&
    stats.bm25_doc_count === 0 &&
    stats.vector_doc_count === 0 &&
    stats.total_queries_processed === 0
  );
}

export default function AnalyticsPanel() {
  const [stats, setStats] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [runs, setRuns] = useState(null);
  const [error, setError] = useState(null);
  const [siblingError, setSiblingError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSiblingError(null);
    // /stats is the panel's backbone — a failure there is the error state;
    // the siblings degrade to a partial view instead of blanking the panel.
    const [statsRes, feedbackRes, runsRes] = await Promise.allSettled([
      fetchJson('/stats'),
      fetchJson('/feedback/summary'),
      fetchJson('/eval/runs'),
    ]);
    if (statsRes.status === 'rejected') {
      setError(statsRes.reason.message);
      setLoading(false);
      return;
    }
    setStats(statsRes.value);
    setFeedback(feedbackRes.status === 'fulfilled' ? feedbackRes.value : null);
    setRuns(runsRes.status === 'fulfilled' ? (runsRes.value.runs || []).slice(0, 5) : null);
    if (feedbackRes.status === 'rejected' || runsRes.status === 'rejected') {
      setSiblingError('Feedback summary or run history is temporarily unavailable.');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const empty =
    stats &&
    isQuiet(stats) &&
    (feedback ? feedback.total === 0 : true) &&
    (runs ? runs.length === 0 : true);

  return (
    <div className="min-h-screen flex flex-col relative" data-testid="analytics-panel">
      <header className="axiom-header relative z-10">
        <div className="flex items-center gap-3">
          <span className="axiom-wordmark">
            <span>◆</span> AXIOM
          </span>
          <span className="text-xs text-gray-500 tracking-widest">
            USAGE ANALYTICS
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <Link
            to="/"
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors"
            data-testid="analytics-back-link"
          >
            <ArrowLeft size={13} /> Dashboard
          </Link>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50"
            data-testid="analytics-refresh"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 relative z-10 flex flex-col gap-6">
        {loading && stats === null && (
          <p className="text-xs text-gray-500" data-testid="analytics-loading">
            Loading usage analytics…
          </p>
        )}

        {error && (
          <div
            className="rounded-lg border border-red-500/20 bg-red-500/5 p-4 text-xs text-red-400"
            data-testid="analytics-error"
          >
            Failed to load usage analytics: {error}
          </div>
        )}

        {!loading && !error && siblingError && (
          <div
            className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-400"
            data-testid="analytics-partial"
          >
            {siblingError}
          </div>
        )}

        {!loading && !error && empty && (
          <div
            className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-6"
            data-testid="analytics-empty"
          >
            <div className="flex items-center gap-2 mb-2">
              <Activity size={14} className="text-violet-400" />
              <h2 className="text-sm font-semibold text-gray-200">No activity recorded yet</h2>
            </div>
            <p className="text-xs text-gray-500">
              Ingest documents and run queries on the dashboard — counts, feedback,
              and eval history appear here. Run history covers runs executed against
              this deployment&apos;s PostgreSQL.
            </p>
          </div>
        )}

        {!loading && !error && stats && !empty && (
          <>
            {/* Core counters — the exact /stats contract, one card per field */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4" data-testid="analytics-stats-grid">
              {STAT_CARDS.map(({ key, label }) => (
                <div
                  key={key}
                  className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
                  data-testid={`stat-${key}`}
                >
                  <p className="text-[11px] text-gray-500">{label}</p>
                  <p className="text-xl font-semibold text-gray-100 font-mono mt-1">
                    {formatCount(stats[key])}
                  </p>
                </div>
              ))}
            </div>

            {typeof stats.stub_mode === 'boolean' && (
              <p className="text-[11px] text-gray-500" data-testid="analytics-stub-mode">
                Mode: {stats.stub_mode ? 'Degraded (stub) — check /api/health' : 'Full pipeline'}
              </p>
            )}

            {/* Feedback summary */}
            <div
              className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
              data-testid="analytics-feedback"
            >
              <div className="flex items-center gap-2 mb-3">
                <MessageSquare size={14} className="text-violet-400" />
                <h2 className="text-sm font-semibold text-gray-200">Feedback</h2>
              </div>
              {feedback ? (
                <>
                  <div className="flex gap-6 text-xs text-gray-400 mb-3" data-testid="analytics-feedback-counts">
                    <span>
                      👍 <span className="font-mono text-gray-200">{feedback.counts?.up ?? 0}</span>
                    </span>
                    <span>
                      👎 <span className="font-mono text-gray-200">{feedback.counts?.down ?? 0}</span>
                    </span>
                    <span>
                      total <span className="font-mono text-gray-200">{feedback.total ?? 0}</span>
                    </span>
                  </div>
                  {(feedback.recent || []).length > 0 ? (
                    <ul className="flex flex-col gap-2">
                      {feedback.recent.map((item) => (
                        <li
                          key={item.id}
                          className="text-xs text-gray-400 border-b border-violet-500/5 pb-2"
                          data-testid="analytics-feedback-row"
                        >
                          <span
                            className={`font-mono mr-2 ${item.rating === 1 ? 'text-emerald-400' : 'text-red-400'}`}
                          >
                            {item.rating === 1 ? '+1' : '-1'}
                          </span>
                          {item.comment || <span className="text-gray-600">no comment</span>}
                          <span className="text-gray-600 ml-2 font-mono">{formatStartedAt(item.created_at)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-gray-600" data-testid="analytics-feedback-empty">
                      No feedback recorded yet.
                    </p>
                  )}
                </>
              ) : (
                <p className="text-xs text-gray-600">Feedback summary unavailable.</p>
              )}
            </div>

            {/* Recent eval runs — full history lives at /eval */}
            <div
              className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
              data-testid="analytics-runs"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <ListChecks size={14} className="text-violet-400" />
                  <h2 className="text-sm font-semibold text-gray-200">Recent eval runs</h2>
                </div>
                <Link
                  to="/eval"
                  className="text-[11px] text-violet-400 hover:text-violet-300 transition-colors"
                  data-testid="analytics-eval-link"
                >
                  Full history →
                </Link>
              </div>
              {runs && runs.length > 0 ? (
                <ul className="flex flex-col gap-2">
                  {runs.map((run) => (
                    <li
                      key={run.job_id}
                      className="flex items-center gap-3 text-xs text-gray-400 border-b border-violet-500/5 pb-2"
                      data-testid="analytics-run-row"
                    >
                      <span className="font-mono text-[11px]">{run.job_id}</span>
                      <span className={run.status === 'complete' ? 'text-emerald-400' : 'text-gray-500'}>
                        {run.status || '—'}
                      </span>
                      <span className="ml-auto text-gray-600">{formatStartedAt(run.started_at)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-600" data-testid="analytics-runs-empty">
                  No eval runs recorded yet.
                </p>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
