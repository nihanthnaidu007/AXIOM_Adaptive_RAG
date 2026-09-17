import React, { useEffect, useState } from 'react';
import { FileText, ChevronDown, ChevronRight, Link2 } from 'lucide-react';
import { fetchJson } from '../../lib/api';

/**
 * Chunk-level citations (Wave 2, D1).
 *
 * Live path: renders citations straight from `reranked_chunks` on the query
 * response / SSE done event — no persistence required. History replay: when
 * the result carries no citations but a trace id exists, the persisted
 * trace_data enrichment is loaded from GET /api/citations/{trace_id}.
 *
 * Click-to-inspect expands a citation to the exact chunk content it backs.
 */
const CONTENT_EXCERPT_LIMIT = 1200;

function normalizeCitations(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((c) => c && typeof c === 'object')
    .map((c, i) => ({
      chunk_id: c.chunk_id || `chunk-${i + 1}`,
      content: typeof c.content === 'string' ? c.content : '',
      source: c.source || c.document_name || 'Unknown source',
      score: typeof c.rerank_score === 'number' ? c.rerank_score : (typeof c.score === 'number' ? c.score : null),
      position: typeof c.position === 'number' ? c.position : i + 1,
    }));
}

export default function CitationsPanel({ citations, traceId, className = '' }) {
  const liveCitations = normalizeCitations(citations);
  const hasLive = liveCitations.length > 0;

  const [history, setHistory] = useState(null);
  const [historyError, setHistoryError] = useState(null);
  const [openChunkId, setOpenChunkId] = useState(null);

  // History replay only when the live result carries no citations — the
  // done payload is authoritative for the just-answered query.
  useEffect(() => {
    if (hasLive || !traceId) {
      setHistory(null);
      setHistoryError(null);
      return;
    }
    let cancelled = false;
    setHistory(null);
    setHistoryError(null);
    fetchJson(`/citations/${encodeURIComponent(traceId)}`)
      .then((payload) => {
        if (!cancelled) setHistory(normalizeCitations(payload.citations));
      })
      .catch((error) => {
        if (!cancelled) setHistoryError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [hasLive, traceId]);

  const shown = hasLive ? liveCitations : history || [];
  const loading = !hasLive && !history && !historyError && Boolean(traceId);

  return (
    <div
      className={`rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4 ${className}`}
      data-testid="citations-panel"
    >
      <div className="flex items-center gap-2 mb-3">
        <Link2 size={14} className="text-violet-400" />
        <h3 className="text-sm font-semibold text-gray-200">
          Citations
        </h3>
        {shown.length > 0 && (
          <span
            className="text-[10px] text-gray-500 border border-gray-700 rounded px-1.5 py-0.5"
            data-testid="citations-count"
          >
            {shown.length}
          </span>
        )}
        {!hasLive && shown.length > 0 && (
          <span className="text-[10px] text-gray-600 italic">replayed from trace</span>
        )}
      </div>

      {shown.length === 0 && !loading && !historyError && (
        <p className="text-xs text-gray-500" data-testid="citations-empty">
          No citations for this query yet.
        </p>
      )}

      {loading && (
        <p className="text-xs text-gray-500" data-testid="citations-loading">
          Loading citations from trace…
        </p>
      )}

      {historyError && (
        <p className="text-xs text-red-400" data-testid="citations-error">
          {historyError}
        </p>
      )}

      <ul className="flex flex-col gap-1">
        {shown.map((citation) => {
          const isOpen = openChunkId === citation.chunk_id;
          return (
            <li key={citation.chunk_id} data-testid="citation-item">
              <button
                type="button"
                onClick={() => setOpenChunkId(isOpen ? null : citation.chunk_id)}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left text-xs text-gray-300 hover:bg-violet-500/10 transition-colors"
                aria-expanded={isOpen}
              >
                {isOpen ? (
                  <ChevronDown size={12} className="text-violet-400 shrink-0" />
                ) : (
                  <ChevronRight size={12} className="text-gray-500 shrink-0" />
                )}
                <FileText size={12} className="text-gray-500 shrink-0" />
                <span className="font-medium text-gray-200 truncate">{citation.source}</span>
                <span className="text-gray-600 shrink-0">
                  chunk {citation.position}
                </span>
                {citation.score != null && (
                  <span className="ml-auto font-mono text-[10px] text-violet-300 shrink-0">
                    {citation.score.toFixed(2)}
                  </span>
                )}
              </button>
              {isOpen && (
                <div
                  className="mt-1 mb-2 mx-8 p-2 rounded bg-black/30 border border-violet-500/10 text-[11px] leading-relaxed text-gray-400 whitespace-pre-wrap break-words"
                  data-testid="citation-detail"
                >
                  {citation.content
                    ? citation.content.slice(0, CONTENT_EXCERPT_LIMIT)
                    : 'No stored content for this chunk.'}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
