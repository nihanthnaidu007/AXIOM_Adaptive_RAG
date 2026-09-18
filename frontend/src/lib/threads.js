/**
 * Browser-local chat-thread registry (Wave 2, A3).
 *
 * The server persists each session's conversation (LangGraph checkpointer
 * keyed by thread_id = session_id) but exposes no enumeration or metadata
 * endpoints — the Wave-2 pre-flight tags full server-side thread management
 * as out of scope. These notes therefore live in this browser's
 * localStorage: rename/archive/delete/search manage the LIBRARY, while
 * resume hands the session id back to the dashboard so the next query
 * continues that thread server-side. Deleting a note never touches server
 * trace retention — the ThreadLibrary panel says so in plain text.
 */

const REGISTRY_KEY = 'axiom_threads_v1';

function readRegistry() {
  try {
    const raw = window.localStorage.getItem(REGISTRY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // Corrupt or unavailable storage — degrade to an empty library rather
    // than break the dashboard; the chat itself never depends on this.
    return [];
  }
}

function writeRegistry(threads) {
  try {
    window.localStorage.setItem(REGISTRY_KEY, JSON.stringify(threads));
  } catch {
    // Storage full or blocked — thread notes are best-effort.
  }
}

/** Derive a display title from the thread's first query. */
export function titleFromQuery(queryText) {
  const trimmed = (queryText || '').trim().replace(/\s+/g, ' ');
  if (!trimmed) return 'Untitled thread';
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}

/**
 * Upsert a thread after a completed query: first sight records it with the
 * query as its title, later sights bump lastUsedAt. Server-backed fact
 * (the session id exists and continues server-side), local note metadata.
 */
export function recordThread(sessionId, queryText) {
  if (!sessionId) return;
  const threads = readRegistry();
  const now = new Date().toISOString();
  const existing = threads.find((t) => t.id === sessionId);
  const next = existing
    ? threads.map((t) =>
        t.id === sessionId ? { ...t, lastUsedAt: now, queryCount: (t.queryCount || 0) + 1 } : t
      )
    : [
        ...threads,
        {
          id: sessionId,
          title: titleFromQuery(queryText),
          archived: false,
          createdAt: now,
          lastUsedAt: now,
          queryCount: 1,
        },
      ];
  writeRegistry(next);
}

/** Threads newest-first; archived threads stay listed but flagged. */
export function listThreads() {
  return readRegistry().sort((a, b) => (a.lastUsedAt < b.lastUsedAt ? 1 : -1));
}

/** Rename a thread's local title; empty titles are ignored. */
export function renameThread(id, title) {
  const trimmed = (title || '').trim();
  if (!trimmed) return;
  writeRegistry(
    readRegistry().map((t) => (t.id === id ? { ...t, title: trimmed.slice(0, 120) } : t))
  );
}

export function setThreadArchived(id, archived) {
  writeRegistry(
    readRegistry().map((t) => (t.id === id ? { ...t, archived: Boolean(archived) } : t))
  );
}

/**
 * Remove the thread's local note. Server-side chat history for the session
 * is NOT deleted — no endpoint exists for that; the panel states this.
 */
export function deleteThread(id) {
  writeRegistry(readRegistry().filter((t) => t.id !== id));
}

/** Case-insensitive title search over a thread list (pure). */
export function searchThreads(threads, query) {
  const needle = (query || '').trim().toLowerCase();
  if (!needle) return threads;
  return threads.filter((t) => (t.title || '').toLowerCase().includes(needle));
}
