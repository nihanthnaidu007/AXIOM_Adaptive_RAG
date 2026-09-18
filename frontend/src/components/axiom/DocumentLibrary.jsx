import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { RefreshCw, FileText, ArrowLeft, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import { fetchJson } from '../../lib/api';

/**
 * Document library (Wave 2, A2).
 *
 * Lists the ingested corpus from GET /api/documents and wires the
 * previously-stranded DELETE /api/documents/{doc_id} behind an inline
 * confirmation. Deleting purges chunk embeddings (pgvector), the BM25
 * index, and lineage server-side, and clears the semantic cache — the
 * panel reloads after a delete so counts never lie.
 *
 * Component-local state only, matching the EvalDashboard pattern:
 * state triple + useCallback loader + the four off-happy-path states.
 */

function formatIndexedAt(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
}

export default function DocumentLibrary() {
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [confirmingId, setConfirmingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchJson('/documents');
      setDocs(payload.documents || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleDelete = useCallback(async (doc) => {
    setDeletingId(doc.doc_id);
    try {
      await fetchJson(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: 'DELETE' });
      toast.success(`Deleted ${doc.filename}`);
      await load();
    } catch (err) {
      toast.error(`Delete failed: ${err.message}`);
    } finally {
      // Release the confirmation on both paths — a failed delete drops back
      // to the row so the user can retry it.
      setConfirmingId(null);
      setDeletingId(null);
    }
  }, [load]);

  return (
    <div className="min-h-screen flex flex-col relative" data-testid="document-library">
      <header className="axiom-header relative z-10">
        <div className="flex items-center gap-3">
          <span className="axiom-wordmark">
            <span>◆</span> AXIOM
          </span>
          <span className="text-xs text-gray-500 tracking-widest">
            DOCUMENT LIBRARY
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <Link
            to="/"
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors"
            data-testid="library-back-link"
          >
            <ArrowLeft size={13} /> Dashboard
          </Link>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1 text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50"
            data-testid="library-refresh"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 relative z-10 flex flex-col gap-6">
        {loading && docs === null && (
          <p className="text-xs text-gray-500" data-testid="library-loading">
            Loading documents…
          </p>
        )}

        {error && (
          <div
            className="rounded-lg border border-red-500/20 bg-red-500/5 p-4 text-xs text-red-400"
            data-testid="library-error"
          >
            Failed to load documents: {error}
          </div>
        )}

        {!loading && !error && docs && docs.length === 0 && (
          <div
            className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-6"
            data-testid="library-empty"
          >
            <div className="flex items-center gap-2 mb-2">
              <FileText size={14} className="text-violet-400" />
              <h2 className="text-sm font-semibold text-gray-200">No documents indexed yet</h2>
            </div>
            <p className="text-xs text-gray-500">
              Upload PDF, TXT, or MD files from the dashboard — every ingested document
              appears here with its chunk count and lineage.
            </p>
          </div>
        )}

        {!loading && !error && docs && docs.length > 0 && (
          <div
            className="rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4"
            data-testid="library-table"
          >
            <h2 className="text-sm font-semibold text-gray-200 mb-3">
              Indexed documents <span className="text-gray-500 font-normal">({docs.length})</span>
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-gray-300" data-testid="library-rows">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-violet-500/10">
                    <th className="py-2 pr-4 font-medium">Document</th>
                    <th className="py-2 pr-4 font-medium">Chunks</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">Indexed</th>
                    <th className="py-2 pr-4 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc) => (
                    <tr
                      key={doc.doc_id}
                      className="border-b border-violet-500/5"
                      data-testid={`library-doc-row-${doc.doc_id}`}
                    >
                      <td className="py-2 pr-4" title={doc.doc_id}>
                        {doc.filename}
                      </td>
                      <td className="py-2 pr-4 font-mono">{doc.chunk_count ?? '—'}</td>
                      <td className="py-2 pr-4">{doc.status || 'indexed'}</td>
                      <td className="py-2 pr-4 text-gray-500">{formatIndexedAt(doc.indexed_at)}</td>
                      <td className="py-2 pr-4 text-right">
                        {confirmingId === doc.doc_id ? (
                          <span className="inline-flex items-center gap-2">
                            <span className="text-gray-500">Delete this document and all of its chunks?</span>
                            <button
                              type="button"
                              onClick={() => handleDelete(doc)}
                              disabled={deletingId === doc.doc_id}
                              className="text-red-400 hover:text-red-300 border border-red-500/30 rounded px-2 py-0.5 disabled:opacity-50"
                              data-testid={`library-confirm-delete-${doc.doc_id}`}
                            >
                              {deletingId === doc.doc_id ? 'Deleting…' : 'Confirm'}
                            </button>
                            <button
                              type="button"
                              onClick={() => setConfirmingId(null)}
                              className="text-gray-400 hover:text-gray-200"
                              data-testid={`library-cancel-delete-${doc.doc_id}`}
                            >
                              Keep
                            </button>
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => setConfirmingId(doc.doc_id)}
                            className="inline-flex items-center gap-1 text-gray-400 hover:text-red-400 transition-colors"
                            data-testid={`library-delete-${doc.doc_id}`}
                          >
                            <Trash2 size={12} /> Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-gray-600 mt-3">
              Deleting removes the document&apos;s chunk embeddings, BM25 index entries,
              lineage record, and invalidates the semantic cache. Chat history is unaffected.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
