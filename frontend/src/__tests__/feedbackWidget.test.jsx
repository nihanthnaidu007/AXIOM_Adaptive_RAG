import React from 'react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import FeedbackWidget from '../components/axiom/FeedbackWidget';

const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());

describe('FeedbackWidget', () => {
  it('renders nothing without a trace id', () => {
    const { container } = render(<FeedbackWidget traceId={null} />);
    expect(container.querySelector('[data-testid="feedback-widget"]')).toBeNull();
  });

  it('posts the chosen rating, trace id, and optional comment', async () => {
    const requests = [];
    server.use(
      http.post('http://127.0.0.1:8000/api/feedback', async ({ request }) => {
        requests.push({
          body: await request.json(),
          apiKey: request.headers.get('X-API-Key'),
        });
        return HttpResponse.json({ id: 1, rating: 1, comment: null });
      })
    );

    render(<FeedbackWidget traceId="trace-9" />);

    fireEvent.click(screen.getByLabelText('Thumbs up'));
    fireEvent.change(screen.getByTestId('feedback-comment'), {
      target: { value: 'Answer cited the right chunk' },
    });
    fireEvent.click(screen.getByTestId('feedback-submit'));

    await waitFor(() => expect(screen.getByTestId('feedback-submitted')).toBeTruthy());
    expect(requests.length).toBe(1);
    expect(requests[0].body).toEqual({
      trace_id: 'trace-9',
      rating: 1,
      comment: 'Answer cited the right chunk',
    });
    // The widget attaches the API key when the build provides one (unset here)
    expect(requests[0].apiKey).toBeNull();
  });

  it('submits a negative rating without a comment', async () => {
    const requests = [];
    server.use(
      http.post('http://127.0.0.1:8000/api/feedback', async ({ request }) => {
        requests.push(await request.json());
        return HttpResponse.json({ id: 2, rating: -1, comment: null });
      })
    );

    render(<FeedbackWidget traceId="trace-9" />);

    fireEvent.click(screen.getByLabelText('Thumbs down'));
    fireEvent.click(screen.getByTestId('feedback-submit'));

    await waitFor(() => expect(screen.getByTestId('feedback-submitted')).toBeTruthy());
    expect(requests[0]).toEqual({ trace_id: 'trace-9', rating: -1, comment: null });
  });

  it('surfaces the sanitized backend error instead of submitting', async () => {
    server.use(
      http.post('http://127.0.0.1:8000/api/feedback', () =>
        HttpResponse.json(
          { detail: { error: 'Unknown trace for feedback' } },
          { status: 400 }
        )
      )
    );

    render(<FeedbackWidget traceId="trace-404" />);

    fireEvent.click(screen.getByLabelText('Thumbs up'));
    fireEvent.click(screen.getByTestId('feedback-submit'));

    await waitFor(() =>
      expect(screen.getByTestId('feedback-error').textContent).toContain('Unknown trace')
    );
    // The form stays visible so the user can retry
    expect(screen.getByTestId('feedback-submit')).toBeTruthy();
    expect(screen.queryByTestId('feedback-submitted')).toBeNull();
  });
});
