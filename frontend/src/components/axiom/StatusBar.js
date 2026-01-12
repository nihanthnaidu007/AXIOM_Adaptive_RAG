import React from 'react';

export const StatusBar = ({ stats, isProcessing, result }) => {
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
      </div>
      
      <div className="flex items-center gap-2">
        {result?.langsmith_trace_url ? (
          <a
            href={result.langsmith_trace_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-violet-400 hover:text-violet-300 transition-colors font-mono text-xs"
          >
            ◆ view trace
          </a>
        ) : (
          <>
            <span className="text-violet-400">◆</span>
            <span className="font-display tracking-wider">AXIOM v1.0</span>
          </>
        )}
        {stats?.stub_mode && (
          <span className="text-[10px] text-amber-400/60 ml-2">(stub mode)</span>
        )}
      </div>
    </div>
  );
};

export default StatusBar;
