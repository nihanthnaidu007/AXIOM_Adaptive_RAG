import React, { useState, useRef, useCallback } from 'react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8000';
const API = `${BACKEND_URL}/api`;

const ACCEPTED_EXTENSIONS = ['.pdf', '.txt', '.md'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getExtension(name) {
  const idx = name.lastIndexOf('.');
  return idx === -1 ? '' : name.slice(idx).toLowerCase();
}

export default function UploadPanel({ onDocsUpdated }) {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const updateFile = useCallback((name, patch) => {
    setUploadedFiles((prev) =>
      prev.map((f) => (f.name === name ? { ...f, ...patch } : f))
    );
  }, []);

  const uploadFile = useCallback(
    async (file) => {
      const ext = getExtension(file.name);
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setUploadedFiles((prev) => [
          ...prev,
          { name: file.name, size: file.size, status: 'failed', chunkCount: 0, error: `Unsupported type: ${ext}` },
        ]);
        return;
      }
      if (file.size > MAX_FILE_SIZE) {
        setUploadedFiles((prev) => [
          ...prev,
          { name: file.name, size: file.size, status: 'failed', chunkCount: 0, error: 'File too large (max 50 MB)' },
        ]);
        return;
      }
      if (file.size === 0) {
        setUploadedFiles((prev) => [
          ...prev,
          { name: file.name, size: file.size, status: 'failed', chunkCount: 0, error: 'File is empty' },
        ]);
        return;
      }

      setUploadedFiles((prev) => [
        ...prev,
        { name: file.name, size: file.size, status: 'uploading', chunkCount: 0, error: null },
      ]);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API}/ingest`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: { error: 'Upload failed' } }));
          throw new Error(err.detail?.error || err.detail || 'Upload failed');
        }

        const data = await response.json();
        updateFile(file.name, {
          status: 'indexed',
          chunkCount: data.chunk_count || 0,
        });
        onDocsUpdated?.();
      } catch (err) {
        updateFile(file.name, {
          status: 'failed',
          error: err.message || 'Upload failed',
        });
      }
    },
    [onDocsUpdated, updateFile]
  );

  const processFiles = useCallback(
    (fileList) => {
      Array.from(fileList).forEach((file) => uploadFile(file));
    },
    [uploadFile]
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

  const clearFiles = () => setUploadedFiles([]);

  const isUploading = uploadedFiles.some((f) => f.status === 'uploading');

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

      {uploadedFiles.length > 0 && (
        <>
          <div className="upload-file-list">
            {uploadedFiles.map((file, i) => (
              <div className="upload-file-row" key={`${file.name}-${i}`}>
                <span className="upload-file-name" title={file.name}>{file.name}</span>
                <span className="upload-file-size">{formatFileSize(file.size)}</span>
                <span className={`upload-status-badge ${file.status}`}>
                  {file.status === 'uploading' && <span className="upload-spinner" />}
                  {file.status === 'uploading' && ' uploading'}
                  {file.status === 'indexed' && 'indexed'}
                  {file.status === 'failed' && 'failed'}
                </span>
                <span className="upload-chunk-count">
                  {file.status === 'indexed' && `${file.chunkCount} chunks`}
                  {file.status === 'failed' && (
                    <span title={file.error} style={{ color: '#fca5a5', cursor: 'help' }}>
                      {file.error?.length > 20 ? file.error.slice(0, 20) + '…' : file.error}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
          <button className="upload-clear-btn" onClick={clearFiles} disabled={isUploading}>
            {isUploading ? 'Uploading…' : 'Clear list'}
          </button>
        </>
      )}
    </div>
  );
}
