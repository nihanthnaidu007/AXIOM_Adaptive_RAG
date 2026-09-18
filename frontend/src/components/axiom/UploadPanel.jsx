import React, { useState, useRef, useCallback } from 'react';
import { API_BASE_URL } from '../../config';
import { authHeaders } from '../../lib/api';

const API = `${API_BASE_URL}/api`;

const ACCEPTED_EXTENSIONS = ['.pdf', '.txt', '.md'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

/**
 * Per-file upload states, mirroring the W4 connector-run registry's
 * pending/running/completed/error vocabulary. POST /api/ingest is
 * synchronous (no job id, nothing to poll) — "running" IS the in-flight
 * request, so progress is response-driven and cancel aborts the fetch.
 * Statuses: queued | running | done | failed | canceled.
 */

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getExtension(name) {
  const idx = name.lastIndexOf('.');
  return idx === -1 ? '' : name.slice(idx).toLowerCase();
}

/** Client-side pre-validation, applied per attempt on every run. */
function validateFile(file) {
  const ext = getExtension(file.name);
  if (!ACCEPTED_EXTENSIONS.includes(ext)) return `Unsupported type: ${ext}`;
  if (file.size > MAX_FILE_SIZE) return 'File too large (max 50 MB)';
  if (file.size === 0) return 'File is empty';
  return null;
}

export default function UploadPanel({ onDocsUpdated }) {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Upload queue: one /ingest in flight at a time — the endpoint is
  // semaphore- and rate-limit-guarded (429/503 under bursts), and a
  // sequential queue is what gives "queued" rows meaning.
  const rowIdRef = useRef(0);
  const chainRef = useRef(Promise.resolve());
  const abortsRef = useRef(new Map()); // row id -> AbortController (running rows)
  const canceledRef = useRef(new Set()); // queued rows canceled before start

  const patchRow = useCallback((id, patch) => {
    setFiles((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }, []);

  const runJob = useCallback(
    async (id, file) => {
      if (canceledRef.current.has(id)) {
        canceledRef.current.delete(id);
        return;
      }
      patchRow(id, { status: 'running' });
      const controller = new AbortController();
      abortsRef.current.set(id, controller);
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API}/ingest`, {
          method: 'POST',
          // /ingest is API-key guarded (fail-closed 401/503) — the same
          // authHeaders every other call attaches. No Content-Type here:
          // the browser sets the multipart boundary for FormData.
          headers: authHeaders(),
          body: formData,
          signal: controller.signal,
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: { error: 'Upload failed' } }));
          throw new Error(err.detail?.error || err.detail || 'Upload failed');
        }

        const data = await response.json();
        patchRow(id, { status: 'done', chunkCount: data.chunk_count || 0, error: null });
        onDocsUpdated?.();
      } catch (err) {
        if (err && err.name === 'AbortError') {
          // User-initiated cancel: a terminal state of its own, never
          // "failed" — the server did nothing wrong and may not have
          // finished processing; nothing is applied after abort.
          patchRow(id, { status: 'canceled', error: null });
        } else {
          patchRow(id, { status: 'failed', error: err?.message || 'Upload failed' });
        }
      } finally {
        abortsRef.current.delete(id);
      }
    },
    [onDocsUpdated, patchRow]
  );

  /** Enqueue behind the previous job; a settled predecessor never rejects. */
  const enqueueJob = useCallback(
    (id, file) => {
      chainRef.current = chainRef.current.then(() => runJob(id, file));
    },
    [runJob]
  );

  const processFiles = useCallback(
    (fileList) => {
      Array.from(fileList).forEach((file) => {
        const failure = validateFile(file);
        rowIdRef.current += 1;
        const id = rowIdRef.current;
        setFiles((prev) => [
          ...prev,
          {
            id,
            file,
            name: file.name,
            size: file.size,
            status: failure ? 'failed' : 'queued',
            chunkCount: null,
            error: failure,
          },
        ]);
        if (!failure) enqueueJob(id, file);
      });
    },
    [enqueueJob]
  );

  const retryRow = useCallback(
    (row) => {
      canceledRef.current.delete(row.id);
      patchRow(row.id, { status: 'queued', error: null, chunkCount: null });
      enqueueJob(row.id, row.file);
    },
    [enqueueJob, patchRow]
  );

  const cancelRow = useCallback(
    (row) => {
      const controller = abortsRef.current.get(row.id);
      if (controller) {
        controller.abort(); // running: fetch rejects with AbortError above
      } else {
        canceledRef.current.add(row.id); // queued: the runJob skip-path
        patchRow(row.id, { status: 'canceled', error: null });
      }
    },
    [patchRow]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
    },
    [processFiles]
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleClick = () => fileInputRef.current?.click();

  const handleFileSelect = useCallback(
    (e) => {
      if (e.target.files.length) processFiles(e.target.files);
      e.target.value = '';
    },
    [processFiles]
  );

  const isBusy = files.some((f) => f.status === 'queued' || f.status === 'running');

  const clearFiles = () => {
    files.forEach((row) => {
      if (row.status === 'queued' || row.status === 'running') cancelRow(row);
    });
    setFiles([]);
  };

  return (
    <div className="upload-panel">
      <div className="upload-panel-header">DOCUMENT UPLOAD</div>

      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && handleClick()}
      >
        <svg className="drop-zone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <div className="drop-zone-text">
          Drop PDF, TXT, or MD files here or <span>click to browse</span>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
      </div>

      {files.length > 0 && (
        <>
          <div className="upload-file-list">
            {files.map((row) => (
              <div className="upload-file-row" key={row.id} data-testid={`upload-row-${row.name}`}>
                <span className="upload-file-name" title={row.name}>{row.name}</span>
                <span className="upload-file-size">{formatFileSize(row.size)}</span>
                <span className={`upload-status-badge ${row.status}`} data-testid={`upload-status-${row.name}`}>
                  {row.status === 'running' && <span className="upload-spinner" />}
                  {row.status === 'running' && ' '}
                  {row.status}
                </span>
                <span className="upload-chunk-count">
                  {row.status === 'done' && row.chunkCount != null && `${row.chunkCount} chunks`}
                  {row.status === 'failed' && row.error && (
                    <span title={row.error} style={{ color: '#fca5a5', cursor: 'help' }}>
                      {row.error.length > 20 ? row.error.slice(0, 20) + '…' : row.error}
                    </span>
                  )}
                </span>
                <span className="upload-row-actions">
                  {(row.status === 'failed' || row.status === 'canceled') && (
                    <button
                      type="button"
                      className="upload-row-btn"
                      data-testid={`upload-retry-${row.name}`}
                      onClick={() => retryRow(row)}
                    >
                      retry
                    </button>
                  )}
                  {(row.status === 'queued' || row.status === 'running') && (
                    <button
                      type="button"
                      className="upload-row-btn"
                      data-testid={`upload-cancel-${row.name}`}
                      onClick={() => cancelRow(row)}
                    >
                      cancel
                    </button>
                  )}
                </span>
              </div>
            ))}
          </div>
          <button className="upload-clear-btn" onClick={clearFiles} disabled={isBusy}>
            {isBusy ? 'Uploading…' : 'Clear list'}
          </button>
        </>
      )}
    </div>
  );
}
