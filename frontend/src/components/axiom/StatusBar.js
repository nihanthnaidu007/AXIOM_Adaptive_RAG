import React from 'react';

const HealthPill = ({ label, status }) => {
  const isHealthy =
    status === 'connected' ||
    status === 'loaded' ||
    status === 'tavily' ||
    status === 'claude-haiku' ||
    status === 'ollama';

  const isDegraded =
    status === 'claude-haiku/unreachable' ||
    status === 'ollama/unavailable' ||
    status === 'not_loaded';

  const isOff =
    status === 'not_connected' ||
    status === 'not_configured' ||
    status === 'unknown';

  const colorClass = isHealthy
    ? 'text-emerald-400/70'
    : isDegraded
    ? 'text-amber-400/70'
    : isOff
    ? 'text-gray-600'
    : 'text-gray-500';

  const dot = isHealthy ? '●' : isDegraded ? '◐' : '○';

  const statusLabel = {
    'connected': 'on',
    'loaded': 'on',
    'tavily': 'on',
    'claude-haiku': 'Haiku',
    'ollama': 'Ollama',
    'claude-haiku/unreachable': 'unreachable',
    'ollama/unavailable': 'unavailable',
    'not_loaded': 'off',
    'not_connected': 'off',
    'not_configured': 'off',
    'unknown': '?',
  }[status] || status;

  return (
    <span className={`font-mono text-[10px] ${colorClass}`} title={`${label}: ${status}`}>
      {dot} {label}:{statusLabel}
    </span>
  );
};

export const StatusBar = ({ stats, isProcessing, result, systemHealth }) => {
  return (
    <div className="axiom-status-bar" data-testid="status-bar">
      <div className="flex items-center gap-6">
        <span className="flex items-center">
          <span className={`status-dot ${isProcessing ? 'active' : 'idle'}`} />
          {isProcessing ? 'Processing' : 'Ready'}
        </span>
        <span>
          Docs: <span className="font-mono">{(stats?.bm25_doc_count ?? 0) || (stats?.vector_doc_count ?? 0)}</span>
        </span>
        <span>
          Cache: <span className="font-mono">{stats?.cache_entries ?? 0}</span> entries
          {typeof stats?.cache_hits === 'number' && (
            <span className="text-gray-500 ml-1">
              · <span className="font-mono">{stats.cache_hits}</span> hits
            </span>
          )}
        </span>
        <span>
          Sessions: <span className="font-mono">{stats?.total_queries_processed || 0}</span>
        </span>

        {systemHealth && (
          <span className="flex items-center gap-3 ml-2 pl-3 border-l border-violet-500/10">
            <HealthPill label="eval" status={systemHealth.evaluator} />
            <HealthPill label="web" status={systemHealth.web_search} />
            <HealthPill label="pg" status={systemHealth.pgvector} />
            <HealthPill label="redis" status={systemHealth.redis} />
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <span className="text-violet-400">◆</span>
        <span className="font-display tracking-wider">AXIOM v1.5</span>
      </div>
    </div>
  );
};

export default StatusBar;
