import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, BarChart3, ArrowLeft } from 'lucide-react';
import { fetchJson } from '../../lib/api';

/**
 * Eval dashboard (Wave 2, D3).
 *
 * Renders the PG-backed eval run history from GET /api/eval/runs:
 * deterministic metrics (completion, strategy accuracy, keyword hit rate)
 * first-class, RAGAS composite when a run carries it. Component-local state
 * only; runs are recorded by the eval runner (nightly real RAGAS — the CI
 * regression gate reads GitHub, it does not persist runs).
 */
const METRICS = [
  { key: 'completion_rate', label: 'Completion' },
  { key: 'strategy_accuracy', label: 'Strategy acc.' },
  { key: 'keyword_hit_rate', label: 'Keyword hit' },
];

function metricOf(run, key) {
  return run?.aggregate && typeof run.aggregate[key] === 'number'
    ? run.aggregate[key]
    : null;
}

function compositeOf(run) {
  const composite = run?.aggregate?.avg_composite_score;
  return typeof composite === 'number' ? composite : null;
}

function formatStartedAt(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
}

export default function EvalDashboard() {
  const [runs, setRuns] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchJson('/eval/runs');
      setRuns(payload.runs || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="min-h-screen flex flex-col relative" data-testid="eval-dashboard">
      <header className="axiom-header relative z-10">
        <div className="flex items-center gap-3">
          <span className="axiom-wordmark">
            <span>◆</span> AXIOM
          </span>
          <span className="text-xs text-gray-500 tracking-widest">
            EVAL RUN HISTORY
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <Link
            to="/"
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors"
            data-testid="eval-back-link"
          >
            <ArrowLeft size={13} /> Dashboard
          </Link>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50"
            data-testid="eval-refresh"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 relative z-10 flex flex-col gap-6">
        {loading && runs === null && (
          <p className="text-xs text-gray-500" data-testid="eval-loading">
            Loading eval runs…
          </p>
        )}

        {error && (
          <div
            className="rounded-lg border border-red-500/20 bg-red-500/5 p-4 text-xs text-red-400"
            data-testid="eval-error"
          >
            Failed to load eval runs: {error}
          </div>
        )}

        {!loading && !error && runs && runs.length === 0 && (
          <div
            className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-6"
            data-testid="eval-empty"
          >
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 size={14} className="text-violet-400" />
              <h2 className="text-sm font-semibold text-gray-200">No eval runs recorded yet</h2>
            </div>
            <p className="text-xs text-gray-500">
              Runs appear here once recorded — the nightly real-RAGAS schedule and
              POST /api/eval/run both persist into eval history. The CI regression
              gate (deterministic metrics) runs per-PR and is read on GitHub, not stored.
            </p>
          </div>
        )}

        {!loading && !error && runs && runs.length > 0 && (
          <>
            {/* Score trends — oldest run leftward, one bar row per metric */}
            <div
              className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
              data-testid="eval-trends"
            >
              <h2 className="text-sm font-semibold text-gray-200 mb-3">Score trends</h2>
              <div className="flex flex-col gap-2">
                {METRICS.map((metric) => {
                  const series = [...runs].reverse().map((run) => metricOf(run, metric.key));
                  const latest = series[series.length - 1];
                  return (
                    <div
                      key={metric.key}
                      className="flex items-center gap-2"
                      data-testid={`trend-${metric.key}`}
                    >
                      <span className="w-24 shrink-0 text-[11px] text-gray-500">{metric.label}</span>
                      <div className="flex-1 flex items-end gap-1 h-8">
                        {series.map((value, i) => (
                          <div
                            key={i}
                            title={value == null ? 'n/a' : value.toFixed(3)}
                            className="w-3 rounded-t bg-violet-500/40"
                            style={{
                              height: `${Math.round((value ?? 0) * 100)}%`,
                              minHeight: value == null ? 2 : 3,
                            }}
                          />
                        ))}
                      </div>
                      <span className="w-14 shrink-0 text-right font-mono text-[11px] text-violet-300">
                        {latest == null ? 'n/a' : latest.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
                <div className="flex items-center gap-2">
                  <span className="w-24 shrink-0 text-[11px] text-gray-500">RAGAS composite</span>
                  <span className="text-[11px] text-gray-500" data-testid="trend-composite">
                    {compositeOf(runs[0]) == null
                      ? 'not present'
                      : `latest ${compositeOf(runs[0]).toFixed(2)}`}
                  </span>
                </div>
              </div>
            </div>

            {/* Run history table */}
            <div
              className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
              data-testid="eval-runs-table"
            >
              <h2 className="text-sm font-semibold text-gray-200 mb-3">
                Run history <span className="text-gray-500 font-normal">({runs.length})</span>
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-gray-300" data-testid="eval-rows">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-violet-500/10">
                      <th className="py-2 pr-4 font-medium">Run</th>
                      <th className="py-2 pr-4 font-medium">Status</th>
                      <th className="py-2 pr-4 font-medium">Progress</th>
                      <th className="py-2 pr-4 font-medium">Completion</th>
                      <th className="py-2 pr-4 font-medium">Strategy acc.</th>
                      <th className="py-2 pr-4 font-medium">Keyword hit</th>
                      <th className="py-2 pr-4 font-medium">Started</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr
                        key={run.job_id}
                        className="border-b border-violet-500/5"
                        data-testid="eval-run-row"
                      >
                        <td className="py-2 pr-4 font-mono text-[11px]">{run.job_id}</td>
                        <td className="py-2 pr-4">{run.status || '—'}</td>
                        <td className="py-2 pr-4">
                          {run.progress != null ? `${run.progress}/${run.total ?? '?'}` : '—'}
                        </td>
                        <td className="py-2 pr-4 font-mono">
                          {metricOf(run, 'completion_rate')?.toFixed(2) ?? '—'}
                        </td>
                        <td className="py-2 pr-4 font-mono">
                          {metricOf(run, 'strategy_accuracy')?.toFixed(2) ?? '—'}
                        </td>
                        <td className="py-2 pr-4 font-mono">
                          {metricOf(run, 'keyword_hit_rate')?.toFixed(2) ?? '—'}
                        </td>
                        <td className="py-2 pr-4 text-gray-500">{formatStartedAt(run.started_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
