import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import EvalDashboard from '../components/axiom/EvalDashboard';

const RUNS = [
  {
    job_id: 'run-newest',
    status: 'complete',
    progress: 30,
    total: 30,
    started_at: '2026-09-17T12:00:00+00:00',
    aggregate: {
      completion_rate: 1.0,
      strategy_accuracy: 1.0,
      keyword_hit_rate: 0.73,
      avg_composite_score: 0.87,
    },
  },
  {
    job_id: 'run-oldest',
    status: 'complete',
    progress: 30,
    total: 30,
    started_at: '2026-09-16T12:00:00+00:00',
    aggregate: {
      completion_rate: 0.95,
      strategy_accuracy: 0.9,
      keyword_hit_rate: 0.66,
    },
  },
];

const server = setupServer(
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

const renderDashboard = () =>
  render(
    <MemoryRouter>
      <EvalDashboard />
    </MemoryRouter>
  );

describe('EvalDashboard', () => {
  it('renders run history newest-first with deterministic metrics', async () => {
    renderDashboard();

    await waitFor(() => expect(screen.getByTestId('eval-rows')).toBeTruthy());
    const rows = screen.getAllByTestId('eval-run-row');
    expect(rows.length).toBe(2);
    expect(rows[0].textContent).toContain('run-newest');
    expect(rows[1].textContent).toContain('run-oldest');
    expect(rows[0].textContent).toContain('1.00'); // completion
    expect(rows[0].textContent).toContain('0.73'); // keyword hit
  });

  it('shows trend rows for deterministic metrics and flags composite presence', async () => {
    renderDashboard();

    await waitFor(() => expect(screen.getByTestId('eval-trends')).toBeTruthy());
    expect(screen.getByTestId('trend-completion_rate')).toBeTruthy();
    expect(screen.getByTestId('trend-strategy_accuracy')).toBeTruthy();
    expect(screen.getByTestId('trend-keyword_hit_rate')).toBeTruthy();
    expect(screen.getByTestId('trend-composite').textContent).toContain('0.87');
  });

  it('shows the empty state when no runs are recorded', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/eval/runs', () =>
        HttpResponse.json({ runs: [], count: 0 })
      )
    );

    renderDashboard();

    await waitFor(() => expect(screen.getByTestId('eval-empty')).toBeTruthy());
  });

  it('surfaces the sanitized error on auth failure', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/eval/runs', () =>
        HttpResponse.json(
          { detail: { error: 'Invalid API key' } },
          { status: 401 }
        )
      )
    );

    renderDashboard();

    await waitFor(() =>
      expect(screen.getByTestId('eval-error').textContent).toContain('Invalid API key')
    );
    expect(screen.queryByTestId('eval-rows')).toBeNull();
  });

  it('refreshes on demand', async () => {
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('eval-rows')).toBeTruthy());

    server.use(
      http.get('http://127.0.0.1:8000/api/eval/runs', () =>
        HttpResponse.json({ runs: [RUNS[0]], count: 1 })
      )
    );
    fireEvent.click(screen.getByTestId('eval-refresh'));

    await waitFor(() => expect(screen.getAllByTestId('eval-run-row').length).toBe(1));
  });
});
