import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import UploadPanel from '../components/axiom/UploadPanel';

const API = 'http://127.0.0.1:8000/api';

/**
 * First component tests for UploadPanel (Wave 2, A0).
 *
 * The panel posts the guarded POST /api/ingest; these tests pin the auth
 * contract (X-API-Key attached from the build-time REACT_APP_API_KEY) and
 * the honest failure surface when auth is missing — the P0 bug was that
 * the upload fetch sent no headers at all, so every correctly-configured
 * deployment 401'd on the core ingest journey.
 */

const server = setupServer(
  http.post(`${API}/ingest`, () =>
    HttpResponse.json({
      filename: 'notes.txt',
      chunk_count: 3,
      status: 'indexed',
      doc_id: 'doc_test_1',
      mode: 'real',
      bm25: 'indexed',
      vector: 'indexed',
    })
  ),
  http.get(`${API}/health`, () =>
    HttpResponse.json({ status: 'ok', nodes: [], system_health: {}, stub_mode: false })
  ),
  http.get(`${API}/stats`, () => HttpResponse.json({ total_documents: 1, total_chunks: 3 }))
);

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
  vi.unstubAllEnvs();
});
afterAll(() => server.close());

function makeFile(name, content, type = 'text/plain') {
  return new File([content], name, { type });
}

/** Drive the hidden file input directly — fireEvent per the suite convention. */
async function chooseFiles(files) {
  const input = document.querySelector('input[type="file"]');
  fireEvent.change(input, { target: { files } });
}

describe('UploadPanel (A0: upload auth)', () => {
  it('posts /ingest with the request auth headers (X-API-Key)', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let seenRequest = null;
    server.use(
      http.post(`${API}/ingest`, async ({ request }) => {
        seenRequest = request;
        return HttpResponse.json({
          filename: 'notes.txt',
          chunk_count: 3,
          status: 'indexed',
          doc_id: 'doc_test_1',
        });
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(seenRequest).not.toBeNull());
    expect(seenRequest.headers.get('X-API-Key')).toBe('test-ui-key');
    // Multipart body preserved: no manual Content-Type overrides the boundary.
    expect(seenRequest.headers.get('Content-Type')).toContain('multipart/form-data');
  });

  it('completes the happy path in a correctly-configured deployment (row flips to indexed)', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(screen.getByText(/indexed/i)).toBeInTheDocument());
    expect(screen.getByText(/3 chunks/)).toBeInTheDocument();
  });

  it('surfaces the server 401 envelope on the failed row when no key is configured', async () => {
    // No key at build time — authHeaders() sends no header, the server 401s.
    vi.stubEnv('REACT_APP_API_KEY', '');
    server.use(
      http.post(`${API}/ingest`, () =>
        HttpResponse.json({ detail: { error: 'Invalid API key' } }, { status: 401 })
      )
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    expect(screen.getByTitle('Invalid API key')).toBeInTheDocument();
  });
});
