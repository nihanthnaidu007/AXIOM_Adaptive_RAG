import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import CitationsPanel from '../components/axiom/CitationsPanel';

const LIVE_CITATIONS = [
  {
    chunk_id: 'bm25-doc-0001',
    content: 'BM25 Okapi scales term frequency with the k1 and b parameters.',
    source: 'bm25-scoring.md',
    rerank_score: 0.92,
  },
  {
    chunk_id: 'rrf-doc-0002',
    content: 'Reciprocal Rank Fusion sums 1/(k + rank) across ranked lists with k=60 by default.',
    source: 'reciprocal-rank-fusion.md',
    rerank_score: 0.71,
  },
];

const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());

describe('CitationsPanel — live path', () => {
  it('renders citations from reranked_chunks with source and score', () => {
    render(<CitationsPanel citations={LIVE_CITATIONS} traceId="trace-1" />);

    expect(screen.getByTestId('citations-panel')).toBeTruthy();
    expect(screen.getByTestId('citations-count').textContent).toBe('2');
    expect(screen.getByText('bm25-scoring.md')).toBeTruthy();
    expect(screen.getByText('reciprocal-rank-fusion.md')).toBeTruthy();
    expect(screen.getByText('0.92')).toBeTruthy();
    // No history fetch happens when live citations exist
    expect(screen.queryByTestId('citations-loading')).toBeNull();
  });

  it('click-to-open reveals the exact chunk content a citation backs', () => {
    render(<CitationsPanel citations={LIVE_CITATIONS} traceId="trace-1" />);

    // Closed state: no detail regions on screen
    expect(screen.queryByTestId('citation-detail')).toBeNull();

    fireEvent.click(screen.getByText('bm25-scoring.md'));

    const detail = screen.getByTestId('citation-detail');
    expect(detail.textContent).toContain('k1 and b parameters');
    // Only the clicked citation opens
    expect(screen.getAllByTestId('citation-detail').length).toBe(1);

    // Click again to close
    fireEvent.click(screen.getByText('bm25-scoring.md'));
    expect(screen.queryByTestId('citation-detail')).toBeNull();
  });

  it('shows the empty state when no citations exist', () => {
    render(<CitationsPanel citations={[]} traceId={null} />);

    expect(screen.getByTestId('citations-empty')).toBeTruthy();
  });
});

describe('CitationsPanel — history replay', () => {
  it('fetches persisted citations when the live result has none', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/citations/:traceId', ({ params }) =>
        HttpResponse.json({
          trace_id: params.traceId,
          citations: LIVE_CITATIONS,
        })
      )
    );

    render(<CitationsPanel citations={[]} traceId="trace-42" />);

    await waitFor(() => expect(screen.getByTestId('citations-count')).toBeTruthy());
    expect(screen.getByText('bm25-scoring.md')).toBeTruthy();
    expect(screen.getByText('replayed from trace')).toBeTruthy();
  });

  it('surfaces the sanitized error when the trace cannot be loaded', async () => {
    server.use(
      http.get('http://127.0.0.1:8000/api/citations/:traceId', () =>
        HttpResponse.json(
          { detail: { error: 'No trace found for session trace-42' } },
          { status: 404 }
        )
      )
    );

    render(<CitationsPanel citations={[]} traceId="trace-42" />);

    await waitFor(() =>
      expect(screen.getByTestId('citations-error').textContent).toContain('No trace found')
    );
  });
});
