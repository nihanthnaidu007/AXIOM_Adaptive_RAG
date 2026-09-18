import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { http, HttpResponse, delay } from 'msw';
import { setupServer } from 'msw/node';
import UploadPanel from '../components/axiom/UploadPanel';

const API = 'http://127.0.0.1:8000/api';

/**
 * UploadPanel tests (Wave 2, A0 + A1).
 *
 * A0 pins the auth contract: the panel posts the guarded POST /api/ingest
 * with X-API-Key from the build-time REACT_APP_API_KEY — the P0 bug was a
 * header-less fetch, so every correctly-configured deployment 401'd.
 *
 * A1 pins the per-file lifecycle: queued → running → done/failed/canceled
 * transitions, retry of a failed file, and cancel that stops the in-flight
 * request (or dequeues a queued one) without ever applying a late response.
 */

const INGEST_OK = {
  filename: 'notes.txt',
  chunk_count: 3,
  status: 'indexed',
  doc_id: 'doc_test_1',
  mode: 'real',
  bm25: 'indexed',
  vector: 'indexed',
};

const server = setupServer(
  http.post(`${API}/ingest`, () => HttpResponse.json(INGEST_OK)),
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

const statusOf = (name) => screen.getByTestId(`upload-status-${name}`);

describe('UploadPanel (A0: upload auth)', () => {
  it('posts /ingest with the request auth headers (X-API-Key)', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let seenRequest = null;
    server.use(
      http.post(`${API}/ingest`, async ({ request }) => {
        seenRequest = request;
        return HttpResponse.json(INGEST_OK);
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(seenRequest).not.toBeNull());
    expect(seenRequest.headers.get('X-API-Key')).toBe('test-ui-key');
    // Multipart body preserved: no manual Content-Type overrides the boundary.
    expect(seenRequest.headers.get('Content-Type')).toContain('multipart/form-data');
  });

  it('completes the happy path in a correctly-configured deployment', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('done'));
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

    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('failed'));
    expect(screen.getByTitle('Invalid API key')).toBeInTheDocument();
  });
});

describe('UploadPanel (A1: per-file lifecycle)', () => {
  it('runs a multi-file drop sequentially with queued → running → done transitions', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let postCount = 0;
    server.use(
      http.post(`${API}/ingest`, async ({ request }) => {
        postCount += 1;
        // Deterministic ordering: hold the FIRST request so the second file
        // is observable in its queued state behind it.
        if (postCount === 1) await delay(150);
        const form = await request.formData();
        return HttpResponse.json({ ...INGEST_OK, filename: form.get('file')?.name });
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('a.txt', 'first'), makeFile('b.txt', 'second')]);

    // First file is in flight (150ms hold); the second waits in the queue.
    await waitFor(() => expect(statusOf('a.txt')).toHaveTextContent('running'));
    expect(statusOf('b.txt')).toHaveTextContent('queued');

    await waitFor(() => expect(statusOf('a.txt')).toHaveTextContent('done'));
    await waitFor(() => expect(statusOf('b.txt')).toHaveTextContent('done'));
    expect(postCount).toBe(2);
  });

  it('retries a failed file and completes on the second attempt', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let attempts = 0;
    server.use(
      http.post(`${API}/ingest`, () => {
        attempts += 1;
        if (attempts === 1) {
          return HttpResponse.json({ detail: { error: 'Ingest failed' } }, { status: 500 });
        }
        return HttpResponse.json(INGEST_OK);
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('failed'));
    expect(screen.getByTitle('Ingest failed')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('upload-retry-notes.txt'));

    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('done'));
    expect(attempts).toBe(2);
    expect(screen.getByText(/3 chunks/)).toBeInTheDocument();
  });

  it('cancel stops the in-flight upload and never applies the late response', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    server.use(
      http.post(`${API}/ingest`, async () => {
        await delay(250);
        return HttpResponse.json(INGEST_OK);
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('notes.txt', 'hello world')]);

    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('running'));
    fireEvent.click(screen.getByTestId('upload-cancel-notes.txt'));
    await waitFor(() => expect(statusOf('notes.txt')).toHaveTextContent('canceled'));

    // The delayed server response would arrive now — cancel must hold.
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(statusOf('notes.txt')).toHaveTextContent('canceled');
    expect(screen.getByTestId('upload-retry-notes.txt')).toBeInTheDocument();
  });

  it('cancel on a queued file dequeues it without a request ever firing', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let postCount = 0;
    server.use(
      http.post(`${API}/ingest`, async () => {
        postCount += 1;
        await delay(150);
        return HttpResponse.json(INGEST_OK);
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('a.txt', 'first'), makeFile('b.txt', 'second')]);

    expect(statusOf('b.txt')).toHaveTextContent('queued');
    fireEvent.click(screen.getByTestId('upload-cancel-b.txt'));
    await waitFor(() => expect(statusOf('b.txt')).toHaveTextContent('canceled'));

    // Let the first upload finish; the canceled file must never run.
    await waitFor(() => expect(statusOf('a.txt')).toHaveTextContent('done'), { timeout: 2000 });
    expect(postCount).toBe(1);
  });

  it('rejects an unsupported type client-side without any POST', async () => {
    vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
    let postCount = 0;
    server.use(
      http.post(`${API}/ingest`, () => {
        postCount += 1;
        return HttpResponse.json(INGEST_OK);
      })
    );

    render(<UploadPanel />);
    await chooseFiles([makeFile('virus.exe', 'binary')]);

    await waitFor(() => expect(statusOf('virus.exe')).toHaveTextContent('failed'));
    expect(screen.getByTitle('Unsupported type: .exe')).toBeInTheDocument();
    expect(postCount).toBe(0);
  });
});
