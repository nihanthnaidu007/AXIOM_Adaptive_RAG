import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import AnalyticsPanel from '../components/axiom/AnalyticsPanel';

// Absolute URLs matching config.js's default backend — same pattern as
// evalDashboard.test.jsx.
const STATS = {
  indexed_documents: 12,
  bm25_doc_count: 340,
  vector_doc_count: 340,
  cache_entries: 7,
  cache_hits: 41,
  total_queries_processed: 96,
  stub_mode: false,
};

const FEEDBACK = {
  total: 2,
  counts: { up: 1, down: 1 },
  recent: [
    {
      id: 'fb-1',
      trace_id: 'trace-1',
      rating: 1,
      comment: 'great answer',
      query_snippet: 'what is axiom',
      created_at: '2026-09-18T09:00:00+00:00',
    },
  ],
};

const RUNS = [
  {
    job_id: 'run-newest',
    status: 'complete',
    progress: 30,
    total: 30,
    started_at: '2026-09-17T12:00:00+00:00',
    aggregate: { keyword_hit_rate: 0.73 },
  },
];

const server = setupServer(
  http.get('http://127.0.0.1:8000/api/stats', () => HttpResponse.json(STATS)),
  http.get('http://127.0.0.1:8000/api/feedback/summary', () => HttpResponse.json(FEEDBACK)),
  http.get('http://127.0.0.1:8000/api/eval/runs', () =>
    HttpResponse.json({ runs: RUNS, count: RUNS.length })
  )
);

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());

const renderPanel = () =>
  render(
    <MemoryRouter>
      <AnalyticsPanel />
    </MemoryRouter>
  );

describe('AnalyticsPanel', () => {
  it('renders one card per /stats counter plus feedback and recent runs', async () => {
    renderPanel();

    await waitFor(() => expect(screen.getByTestId('analytics-stats-grid')).toBeTruthy());
    // Six counter cards; stub_mode renders as the mode line, not a card.
    for (const key of ['indexed_documents', 'bm25_doc_count', 'vector_doc_count', 'cache_entries', 'cache_hits', 'total_queries_processed']) {
      expect(screen.getByTestId(`stat-${key}`).textContent).toContain(
        String(STATS[key])
      );
    }
    expect(screen.getByTestId('analytics-stub-mode').textContent).toContain('Full pipeline');
    expect(screen.getByTestId('analytics-feedback-counts').textContent).toContain('2');
    expect(screen.getByTestId('analytics-feedback-row').textContent).toContain('great answer');
    expect(screen.getAllByTestId('analytics-run-row').length).toBe(1);
    expect(screen.getAllByTestId('analytics-run-row')[0].textContent).toContain('run-newest');
  });

  it('shows the empty state when every source reports no activity', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/stats', () =>
        HttpResponse.json({ ...STATS, indexed_documents: 0, bm25_doc_count: 0, vector_doc_count: 0, cache_entries: 0, cache_hits: 0, total_queries_processed: 0 })
      ),
      http.get('http://127.0.0.1:8000/api/feedback/summary', () =>
        HttpResponse.json({ total: 0, counts: { up: 0, down: 0 }, recent: [] })
      ),
      http.get('http://127.0.0.1:8000/api/eval/runs', () =>
        HttpResponse.json({ runs: [], count: 0 })
      )
    );

    renderPanel();

    await waitFor(() => expect(screen.getByTestId('analytics-empty')).toBeTruthy());
    expect(screen.queryByTestId('analytics-stats-grid')).toBeNull();
  });

  it('surfaces the sanitized error when /stats fails closed', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/stats', () =>
        HttpResponse.json({ detail: { error: 'Invalid API key' } }, { status: 401 })
      )
    );

    renderPanel();

    await waitFor(() =>
      expect(screen.getByTestId('analytics-error').textContent).toContain('Invalid API key')
    );
    expect(screen.queryByTestId('analytics-stats-grid')).toBeNull();
  });

  it('degrades to a partial view when a sibling source fails but stats load', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/feedback/summary', () =>
        HttpResponse.json({ detail: { error: 'Upstream unavailable' } }, { status: 500 })
      )
    );

    renderPanel();

    await waitFor(() => expect(screen.getByTestId('analytics-partial')).toBeTruthy());
    expect(screen.getByTestId('analytics-stats-grid')).toBeTruthy();
  });

  it('refreshes on demand', async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('analytics-stats-grid')).toBeTruthy());

    server.use(
      http.get('http://127.0.0.1:8000/api/stats', () =>
        HttpResponse.json({ ...STATS, total_queries_processed: 97 })
      )
    );
    fireEvent.click(screen.getByTestId('analytics-refresh'));

    await waitFor(() =>
      expect(screen.getByTestId('stat-total_queries_processed').textContent).toContain('97')
    );
  });
});
