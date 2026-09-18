import React from 'react';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import ThreadLibrary from '../components/axiom/ThreadLibrary';
import {
  recordThread,
  renameThread,
  titleFromQuery,
  searchThreads,
} from '../lib/threads';

/**
 * Chat thread management tests (Wave 2, A3).
 *
 * The registry is browser-local by design (no server enumeration/mutation
 * endpoints exist — resume is the only server-backed op, handing session_id
 * back to the dashboard). Pinned here: the registry round-trips, the panel's
 * render + search + rename/archive/delete flows, the resume contract, the
 * empty state, and the honesty note.
 */

const REGISTRY_KEY = 'axiom_threads_v1';

const T1 = {
  id: 'aaaaaaaa-1111-2222-3333-444444444444',
  title: 'What is adaptive retrieval?',
  archived: false,
  createdAt: '2026-09-18T09:00:00.000Z',
  lastUsedAt: '2026-09-18T12:00:00.000Z',
  queryCount: 3,
};
const T2 = {
  id: 'bbbbbbbb-5555-6666-7777-888888888888',
  title: 'Explain BM25 indexing',
  archived: false,
  createdAt: '2026-09-18T08:00:00.000Z',
  lastUsedAt: '2026-09-18T11:00:00.000Z',
  queryCount: 1,
};

function seed(threads) {
  window.localStorage.setItem(REGISTRY_KEY, JSON.stringify(threads));
}

function readRegistry() {
  return JSON.parse(window.localStorage.getItem(REGISTRY_KEY) || '[]');
}

function renderPanel(props = {}) {
  const onResume = props.onResume || (() => {});
  const onNewThread = props.onNewThread || (() => {});
  return render(
    <ThreadLibrary
      sessionId={'sessionId' in props ? props.sessionId : 'current-session'}
      onResume={onResume}
      onNewThread={onNewThread}
    />
  );
}

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe('thread registry (lib/threads)', () => {
  it('records a new thread from a completed query and bumps repeats', () => {
    recordThread('sess-1', 'How does the RAG pipeline adapt?');
    expect(readRegistry()).toHaveLength(1);
    expect(readRegistry()[0].title).toBe('How does the RAG pipeline adapt?');
    expect(readRegistry()[0].queryCount).toBe(1);

    recordThread('sess-1', 'Follow-up question');
    expect(readRegistry()).toHaveLength(1);
    expect(readRegistry()[0].queryCount).toBe(2);
    // Title stays the first query's — rename is the only title mutation.
    expect(readRegistry()[0].title).toBe('How does the RAG pipeline adapt?');
  });

  it('ignores empty session ids and blank rename titles', () => {
    recordThread(null, 'orphan query');
    expect(readRegistry()).toHaveLength(0);

    recordThread('sess-1', 'first query');
    renameThread('sess-1', '   ');
    expect(readRegistry()[0].title).toBe('first query');
  });

  it('derives titles: trimmed, whitespace-collapsed, truncated', () => {
    expect(titleFromQuery('  what   is rag? ')).toBe('what is rag?');
    expect(titleFromQuery('x'.repeat(60)).length).toBe(49);
    expect(titleFromQuery('')).toBe('Untitled thread');
  });

  it('searchThreads filters case-insensitively and passes through blank queries', () => {
    const threads = [T1, T2];
    expect(searchThreads(threads, 'bm25')).toEqual([T2]);
    expect(searchThreads(threads, '')).toEqual(threads);
  });
});

describe('ThreadLibrary panel (A3)', () => {
  it('renders recorded threads newest-first with the active badge', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: T1.id });

    const rows = screen.getAllByTestId(/^thread-row-/);
    expect(rows).toHaveLength(2);
    expect(rows[0].dataset.testid).toBe('thread-row-aaaaaaaa');
    expect(screen.getByText('active')).toBeInTheDocument();
  });

  it('search filters the visible rows and shows a no-match state', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: null });

    fireEvent.change(screen.getByTestId('thread-search'), {
      target: { value: 'BM25' },
    });
    expect(screen.queryByTestId('thread-row-aaaaaaaa')).not.toBeInTheDocument();
    expect(screen.getByTestId('thread-row-bbbbbbbb')).toBeInTheDocument();

    fireEvent.change(screen.getByTestId('thread-search'), {
      target: { value: 'no-such-thread' },
    });
    expect(screen.getByTestId('thread-search-empty')).toBeInTheDocument();
  });

  it('renames a thread inline and persists the new title', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: null });

    fireEvent.click(screen.getByTestId('thread-rename-aaaaaaaa'));
    const input = screen.getByTestId('thread-rename-input-aaaaaaaa');
    fireEvent.change(input, { target: { value: 'Retrieval deep dive' } });
    fireEvent.click(screen.getByTestId('thread-rename-save-aaaaaaaa'));

    expect(screen.getByText('Retrieval deep dive')).toBeInTheDocument();
    expect(readRegistry().find((t) => t.id === T1.id).title).toBe('Retrieval deep dive');
  });

  it('archives and unarchives a thread', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: null });

    fireEvent.click(screen.getByTestId('thread-archive-aaaaaaaa'));
    expect(screen.getByText('archived')).toBeInTheDocument();
    expect(readRegistry().find((t) => t.id === T1.id).archived).toBe(true);

    fireEvent.click(screen.getByTestId('thread-archive-aaaaaaaa'));
    expect(screen.queryByText('archived')).not.toBeInTheDocument();
    expect(readRegistry().find((t) => t.id === T1.id).archived).toBe(false);
  });

  it('deletes a thread only through the inline confirmation', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: null });

    fireEvent.click(screen.getByTestId('thread-delete-aaaaaaaa'));
    expect(screen.getByTestId('thread-row-aaaaaaaa')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('thread-confirm-delete-aaaaaaaa'));
    expect(screen.queryByTestId('thread-row-aaaaaaaa')).not.toBeInTheDocument();
    expect(readRegistry().map((t) => t.id)).toEqual([T2.id]);
  });

  it('keeps the row when the delete confirmation is dismissed', () => {
    seed([T1, T2]);
    renderPanel({ sessionId: null });

    fireEvent.click(screen.getByTestId('thread-delete-aaaaaaaa'));
    fireEvent.click(screen.getByTestId('thread-cancel-delete-aaaaaaaa'));

    expect(screen.getByTestId('thread-row-aaaaaaaa')).toBeInTheDocument();
    expect(readRegistry()).toHaveLength(2);
  });

  it('resume hands the full session id back to the dashboard', () => {
    seed([T1, T2]);
    const resumed = [];
    renderPanel({ sessionId: null, onResume: (id) => resumed.push(id) });

    fireEvent.click(screen.getByTestId('thread-resume-bbbbbbbb'));
    expect(resumed).toEqual([T2.id]);
  });

  it('new chat asks the dashboard for a fresh session', () => {
    seed([T1]);
    let newThreadCalls = 0;
    renderPanel({ sessionId: T1.id, onNewThread: () => { newThreadCalls += 1; } });

    fireEvent.click(screen.getByTestId('thread-new'));
    expect(newThreadCalls).toBe(1);
    // The active thread shows no Resume button — you are already in it.
    expect(screen.queryByTestId(`thread-resume-${T1.id.slice(0, 8)}`)).not.toBeInTheDocument();
  });

  it('shows the empty state and the honesty note', () => {
    renderPanel({ sessionId: null });

    expect(screen.getByTestId('thread-empty')).toBeInTheDocument();
    expect(
      screen.getByText(/deleting a note here does not delete server chat history/)
    ).toBeInTheDocument();
  });
});
