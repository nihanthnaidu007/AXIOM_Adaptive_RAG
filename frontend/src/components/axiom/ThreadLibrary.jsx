import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { MessageSquare, Archive, ArchiveRestore, Pencil, Trash2, Check, X, Play } from 'lucide-react';
import {
  listThreads,
  renameThread,
  setThreadArchived,
  deleteThread,
  searchThreads,
} from '../../lib/threads';

/**
 * Chat thread library (Wave 2, A3).
 *
 * A dashboard section over the sessions this browser has seen: rename,
 * archive/unarchive, delete (inline confirmation), and title search.
 * RESUME is server-backed — it sets the active session_id so the next
 * query continues that thread through the LangGraph checkpointer.
 * Rename/archive/delete manage the browser-local note (no server
 * enumeration/mutation endpoints exist — see lib/threads.js); the footer
 * states plainly that deleting a note does not touch server trace
 * retention. Fork is not offered: it needs server-side state copy.
 *
 * Component-local state, matching the panel pattern (loading/error are
 * meaningless for a synchronous local registry, but the empty state is).
 */

function formatRelative(iso) {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.round(diffMs / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function ThreadRow({ thread, isActive, onResume, onMutate }) {
  const [confirming, setConfirming] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draftTitle, setDraftTitle] = useState(thread.title);

  const shortId = thread.id.slice(0, 8);

  return (
    <div
      className={`thread-row ${isActive ? 'thread-row-active' : ''} ${
        thread.archived ? 'thread-row-archived' : ''
      }`}
      data-testid={`thread-row-${shortId}`}
    >
      <span className="thread-row-icon">
        <MessageSquare size={13} />
      </span>

      {renaming ? (
        <span className="thread-row-rename">
          <input
            type="text"
            value={draftTitle}
            onChange={(e) => setDraftTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                renameThread(thread.id, draftTitle);
                onMutate();
                setRenaming(false);
              } else if (e.key === 'Escape') {
                setDraftTitle(thread.title);
                setRenaming(false);
              }
            }}
            autoFocus
            maxLength={120}
            data-testid={`thread-rename-input-${shortId}`}
          />
          <button
            type="button"
            title="Save title"
            onClick={() => {
              renameThread(thread.id, draftTitle);
              onMutate();
              setRenaming(false);
            }}
            data-testid={`thread-rename-save-${shortId}`}
          >
            <Check size={12} />
          </button>
          <button
            type="button"
            title="Discard"
            onClick={() => {
              setDraftTitle(thread.title);
              setRenaming(false);
            }}
            data-testid={`thread-rename-cancel-${shortId}`}
          >
            <X size={12} />
          </button>
        </span>
      ) : (
        <span className="thread-row-main">
          <span className="thread-row-title" title={thread.title}>
            {thread.title}
          </span>
          <span className="thread-row-meta">
            {thread.archived && <span className="thread-badge-archived">archived</span>}
            {isActive && <span className="thread-badge-active">active</span>}
            <span className="thread-row-when">
              {thread.queryCount || 0} q · {formatRelative(thread.lastUsedAt)}
            </span>
          </span>
        </span>
      )}

      {confirming ? (
        <span className="thread-row-confirm">
          <span className="text-gray-500">Delete note?</span>
          <button
            type="button"
            className="thread-btn thread-btn-danger"
            onClick={() => {
              deleteThread(thread.id);
              onMutate();
            }}
            data-testid={`thread-confirm-delete-${shortId}`}
          >
            Confirm
          </button>
          <button
            type="button"
            className="thread-btn"
            onClick={() => setConfirming(false)}
            data-testid={`thread-cancel-delete-${shortId}`}
          >
            Keep
          </button>
        </span>
      ) : (
        <span className="thread-row-actions">
          {!isActive && !thread.archived && (
            <button
              type="button"
              className="thread-btn thread-btn-primary"
              title="Resume: continue this thread in the chat"
              onClick={() => onResume(thread.id)}
              data-testid={`thread-resume-${shortId}`}
            >
              <Play size={11} /> Resume
            </button>
          )}
          <button
            type="button"
            className="thread-btn"
            title="Rename"
            onClick={() => {
              setDraftTitle(thread.title);
              setRenaming(true);
            }}
            data-testid={`thread-rename-${shortId}`}
          >
            <Pencil size={11} />
          </button>
          <button
            type="button"
            className="thread-btn"
            title={thread.archived ? 'Unarchive' : 'Archive'}
            onClick={() => {
              setThreadArchived(thread.id, !thread.archived);
              onMutate();
            }}
            data-testid={`thread-archive-${shortId}`}
          >
            {thread.archived ? <ArchiveRestore size={11} /> : <Archive size={11} />}
          </button>
          <button
            type="button"
            className="thread-btn thread-btn-danger-ghost"
            title="Delete note"
            onClick={() => setConfirming(true)}
            data-testid={`thread-delete-${shortId}`}
          >
            <Trash2 size={11} />
          </button>
        </span>
      )}
    </div>
  );
}

export default function ThreadLibrary({ sessionId, onResume, onNewThread }) {
  const [threads, setThreads] = useState(null);
  const [search, setSearch] = useState('');

  const refresh = useCallback(() => {
    setThreads(listThreads());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const visible = useMemo(
    () => (threads ? searchThreads(threads, search) : []),
    [threads, search]
  );

  return (
    <div className="thread-panel" data-testid="thread-panel">
      <div className="thread-panel-header">
        <span className="thread-panel-title">CHAT THREADS</span>
        <span className="thread-panel-actions">
          <button
            type="button"
            className="thread-btn thread-btn-primary"
            onClick={onNewThread}
            data-testid="thread-new"
          >
            New chat
          </button>
        </span>
      </div>

      {threads && threads.length > 0 && (
        <input
          type="text"
          className="thread-search"
          placeholder="Search threads by title…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          data-testid="thread-search"
        />
      )}

      {threads && threads.length === 0 && (
        <p className="thread-empty" data-testid="thread-empty">
          Threads appear here as you chat — each query starts or continues a
          server-backed session.
        </p>
      )}

      {threads && threads.length > 0 && visible.length === 0 && (
        <p className="thread-empty" data-testid="thread-search-empty">
          No threads match “{search}”.
        </p>
      )}

      {visible.length > 0 && (
        <div className="thread-list" data-testid="thread-list">
          {visible.map((thread) => (
            <ThreadRow
              key={thread.id}
              thread={thread}
              isActive={thread.id === sessionId}
              onResume={onResume}
              onMutate={refresh}
            />
          ))}
        </div>
      )}

      <p className="thread-note">
        Thread notes live in this browser. Resume continues the real conversation
        server-side by session id; deleting a note here does not delete server
        chat history or traces.
      </p>
    </div>
  );
}
