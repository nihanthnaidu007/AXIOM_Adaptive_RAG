import React from 'react';
import { describe, it, expect, beforeAll, beforeEach, afterEach, afterAll, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import DocumentLibrary from '../components/axiom/DocumentLibrary';

const API = 'http://127.0.0.1:8000/api';

/**
 * DocumentLibrary tests (Wave 2, A2).
 *
 * The panel lists the corpus from GET /api/documents and wires the
 * previously-stranded DELETE /api/documents/{doc_id} behind an inline
 * confirmation. Pinned here: the list render (count + rows), the four
 * UI states, the delete contract (right path, list refreshes without the
 * deleted doc), and the auth header on every call.
 */

const DOCS = {
  documents: [
    {
      doc_id: 'doc_aaa111',
      filename: 'handbook.txt',
      chunk_count: 12,
      indexed_at: '2026-09-18T10:00:00+00:00',
      status: 'indexed',
    },
    {
      doc_id: 'doc_bbb222',
      filename: 'notes.md',
      chunk_count: 4,
      indexed_at: '2026-09-18T11:00:00+00:00',
      status: 'indexed',
    },
  ],
  count: 2,
};

function renderLibrary() {
  return render(
    <MemoryRouter>
      <DocumentLibrary />
    </MemoryRouter>
  );
}

const server = setupServer(
  http.get(`${API}/documents`, () => HttpResponse.json(DOCS)),
  http.delete(`${API}/documents/:docId`, ({ params }) =>
    HttpResponse.json({
      doc_id: params.docId,
      filename: 'handbook.txt',
      deleted_chunks: 12,
      cache_keys_cleared: 1,
      status: 'deleted',
    })
  )
);

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
beforeEach(() => {
  vi.stubEnv('REACT_APP_API_KEY', 'test-ui-key');
});
afterEach(() => {
  server.resetHandlers();
  cleanup();
  vi.unstubAllEnvs();
});
afterAll(() => server.close());

describe('DocumentLibrary (A2)', () => {
  it('lists documents from GET /documents with the auth header attached', async () => {
    let seenRequest = null;
    server.use(
      http.get(`${API}/documents`, ({ request }) => {
        seenRequest = request;
        return HttpResponse.json(DOCS);
      })
    );

    renderLibrary();

    await waitFor(() => expect(screen.getByTestId('library-rows')).toBeInTheDocument());
    expect(seenRequest.headers.get('X-API-Key')).toBe('test-ui-key');
    // The count is split across JSX elements — assert on the table's text.
    expect(screen.getByTestId('library-table')).toHaveTextContent('Indexed documents (2)');
    expect(screen.getByText('handbook.txt')).toBeInTheDocument();
    expect(screen.getByText('notes.md')).toBeInTheDocument();
  });

  it('shows the empty state when no documents are indexed', async () => {
    server.use(
      http.get(`${API}/documents`, () => HttpResponse.json({ documents: [], count: 0 }))
    );

    renderLibrary();

    await waitFor(() => expect(screen.getByTestId('library-empty')).toBeInTheDocument());
  });

  it('shows the error state when the listing fails', async () => {
    server.use(
      http.get(
        `${API}/documents`,
        () => HttpResponse.json({ detail: { error: 'Server exploded' } }, { status: 500 })
      )
    );

    renderLibrary();

    await waitFor(() =>
      expect(screen.getByText(/Failed to load documents: Server exploded/)).toBeInTheDocument()
    );
    expect(screen.getByTestId('library-error')).toBeInTheDocument();
  });

  it('deletes through the inline confirmation and refreshes without the doc', async () => {
    let deletePath = null;
    let deleted = false;
    server.use(
      http.get(`${API}/documents`, () =>
        HttpResponse.json(
          deleted ? { documents: [DOCS.documents[1]], count: 1 } : DOCS
        )
      ),
      http.delete(`${API}/documents/:docId`, ({ request, params }) => {
        deleted = true;
        deletePath = new URL(request.url).pathname;
        return HttpResponse.json({
          doc_id: params.docId,
          filename: 'handbook.txt',
          deleted_chunks: 12,
          cache_keys_cleared: 1,
          status: 'deleted',
        });
      })
    );

    renderLibrary();

    await waitFor(() => expect(screen.getByText('handbook.txt')).toBeInTheDocument());

    // Step 1: arm the confirmation — nothing is deleted yet.
    fireEvent.click(screen.getByTestId('library-delete-doc_aaa111'));
    expect(screen.getByTestId('library-confirm-delete-doc_aaa111')).toBeInTheDocument();

    // Step 2: confirm — the DELETE hits the right path and the list refreshes.
    fireEvent.click(screen.getByTestId('library-confirm-delete-doc_aaa111'));

    await waitFor(() => expect(deletePath).toBe('/api/documents/doc_aaa111'));
    await waitFor(() => expect(screen.queryByText('handbook.txt')).not.toBeInTheDocument());
    expect(screen.getByTestId('library-table')).toHaveTextContent('Indexed documents (1)');
  });

  it('keeps the doc when the confirmation is dismissed', async () => {
    let deleteCalls = 0;
    server.use(
      http.delete(`${API}/documents/:docId`, () => {
        deleteCalls += 1;
        return HttpResponse.json({ status: 'deleted' });
      })
    );

    renderLibrary();

    await waitFor(() => expect(screen.getByText('handbook.txt')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('library-delete-doc_aaa111'));
    fireEvent.click(screen.getByTestId('library-cancel-delete-doc_aaa111'));

    expect(screen.getByText('handbook.txt')).toBeInTheDocument();
    expect(screen.queryByTestId('library-confirm-delete-doc_aaa111')).not.toBeInTheDocument();
    expect(deleteCalls).toBe(0);
  });

  it('surfaces the failure when the delete call errors', async () => {
    server.use(
      http.delete(
        `${API}/documents/:docId`,
        () => HttpResponse.json({ detail: { error: 'Delete failed' } }, { status: 500 })
      )
    );

    renderLibrary();

    await waitFor(() => expect(screen.getByText('handbook.txt')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('library-delete-doc_aaa111'));
    fireEvent.click(screen.getByTestId('library-confirm-delete-doc_aaa111'));

    // The failed delete leaves the row in place (confirmation cleared) so
    // the user can retry — the doc is NOT silently dropped from the list.
    await waitFor(() =>
      expect(screen.queryByTestId('library-confirm-delete-doc_aaa111')).not.toBeInTheDocument()
    );
    expect(screen.getByText('handbook.txt')).toBeInTheDocument();
    expect(screen.getByTestId('library-table')).toHaveTextContent('Indexed documents (2)');
  });
});
