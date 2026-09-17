import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import App from '../App';

const API = 'http://127.0.0.1:8000/api';

/**
 * Build an SSE Response whose body is a ReadableStream emitting `frames`
 * (raw "data: ..." strings) with an optional inter-frame delay, so tests can
 * observe progressive rendering between chunks.
 */
function sseResponse(frames, { delayMs = 0 } = {}) {
  const encoder = new TextEncoder();
  let index = 0;
  const body = new ReadableStream({
    async pull(controller) {
      if (index >= frames.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(frames[index]));
      index += 1;
      if (delayMs) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    },
  });
  return new Response(body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'X-Request-ID': 'test-req-1',
    },
  });
}

const frame = (payload) => `data: ${JSON.stringify(payload)}\n\n`;
const DONE_FRAME = 'data: [DONE]\n\n';

const DONE_RESULT = {
  session_id: 'sess-1',
  final_answer: 'Streaming answer text',
  confidence: { label: 'PROBABLE', score: 0.8, reasoning: 'grounded' },
  classification: { query_type: 'factual', retrieval_strategy: 'hybrid' },
  retrieval_strategy: 'hybrid',
  ragas_scores: { evaluation_mode: 'full', faithfulness: 0.91 },
  scores_history: [],
  reranked_chunks: [
    { chunk_id: 'c1', content: 'chunk one body', source: 'handbook.pdf', rerank_score: 0.91 },
  ],
  correction_attempts: 0,
  correction_history: [],
  trace_steps: [{ node_name: 'generate_answer', status: 'complete', summary: 'ok' }],
  served_from_cache: false,
  is_complete: true,
  error: null,
  total_latency_ms: 12.5,
  web_search_used: false,
  web_search_chunks: [],
  document_chunk_count: 1,
  web_chunk_count: 0,
};

function streamFrames({ withSources = true, withCache = false } = {}) {
  const result = {
    ...DONE_RESULT,
    served_from_cache: withCache,
  };
  const frames = [
    frame({ type: 'status', stage: 'retrieving', request_id: 'test-req-1' }),
    frame({ type: 'status', stage: 'generating', request_id: 'test-req-1' }),
    frame({ type: 'content', delta: 'Streaming ', request_id: 'test-req-1' }),
    frame({ type: 'content', delta: 'answer text', request_id: 'test-req-1' }),
  ];
  if (withSources) {
    frames.push(
      frame({
        type: 'sources',
        sources: [
          { kind: 'document', chunk_id: 'c1', source: 'handbook.pdf', score: 0.91, preview: 'chunk one body' },
        ],
        web_search_used: false,
        request_id: 'test-req-1',
      })
    );
  }
  frames.push(frame({ type: 'done', result, request_id: 'test-req-1' }), DONE_FRAME);
  return frames;
}

const server = setupServer(
  http.get(`${API}/health`, () =>
    HttpResponse.json({ status: 'ok', nodes: [], system_health: {}, stub_mode: false })
  ),
  http.get(`${API}/stats`, () =>
    HttpResponse.json({ total_documents: 1, total_chunks: 3 })
  ),
  http.post(`${API}/query/stream`, () => sseResponse(streamFrames(), { delayMs: 25 }))
);

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());

async function submitQuery() {
  render(<App />);
  await screen.findByTestId('query-input');
  // fireEvent, not userEvent: userEvent's pointer-based click targets (0,0)
  // in jsdom, where the decorative HexBackground overlay swallows it.
  fireEvent.change(screen.getByTestId('query-input'), {
    target: { value: 'what is axiom' },
  });
  fireEvent.click(screen.getByTestId('run-query-btn'));
}

describe('streaming answer UI (D2)', () => {
  it('renders stage signals, then the progressive answer, then citations and the final result', async () => {
    await submitQuery();

    // While the stream is still sending frames, a stage indicator is visible.
    await screen.findByTestId('stream-stage');
    expect(
      screen.getByText(/Retrieving context\.\.\.|Generating answer\.\.\./)
    ).toBeInTheDocument();

    // Progressive answer: the first delta renders before the final result.
    await screen.findByText(/Streaming/);

    // Citations from the "sources" event appear with the final result.
    const citations = await screen.findByTestId('stream-citations');
    await waitFor(() => {
      expect(citations).toHaveTextContent('handbook.pdf');
      expect(citations).toHaveTextContent('0.91');
    });

    // Terminal state: stage indicator gone, full answer, confidence badge.
    // Scope to the answer panel: the completion toast also contains the
    // confidence label, and the final render flushes a beat after the stage
    // indicator clears — findBy* absorbs both.
    await waitFor(() => {
      expect(screen.queryByTestId('stream-stage')).not.toBeInTheDocument();
    });
    const panel = screen.getByTestId('answer-panel');
    await within(panel).findByText('Streaming answer text');
    await within(panel).findByText(/PROBABLE/i);
  });

  it('shows the fail state when the stream reports an in-band error', async () => {
    server.use(
      http.post(`${API}/query/stream`, () =>
        sseResponse(
          [
            frame({ type: 'status', stage: 'retrieving', request_id: 'test-req-1' }),
            frame({
              type: 'error',
              code: 'internal_error',
              message: 'An unexpected error occurred while processing the request. Check server logs for details.',
              request_id: 'test-req-1',
            }),
            DONE_FRAME,
          ],
          { delayMs: 10 }
        )
      )
    );

    await submitQuery();

    await screen.findByText('Query Failed');
    // No terminal done result: the answer area returns to the empty state.
    await waitFor(() => {
      expect(screen.queryByText(/Retrieving context\.\.\./)).not.toBeInTheDocument();
    });
  });

  it('marks the answer panel as a cache hit when the done event says so', async () => {
    server.use(
      http.post(`${API}/query/stream`, () =>
        sseResponse(streamFrames({ withCache: true }), { delayMs: 10 })
      )
    );

    await submitQuery();

    await screen.findByText('Streaming answer text');
    await screen.findByText('CACHE HIT');
  });
});
