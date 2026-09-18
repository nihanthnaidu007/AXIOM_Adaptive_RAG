import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnswerPanel } from '../components/axiom/AnswerPanel';

/**
 * LangSmith trace-link tests (Wave 2, A4).
 *
 * The backend returns `langsmith_trace_url` in QueryResponse only when
 * LangSmith tracing is enabled (server.py populates it from
 * langsmith_tracer.get_trace_url; it is Optional[str] = None otherwise).
 * Pinned: the link renders when present, is absent when absent — no dead
 * link, no fake trace UI.
 */

const BASE_PROPS = {
  answer: 'BM25 scores documents using term frequency.',
  confidence: { label: 'VERIFIED', score: 0.89 },
  isLoading: false,
  servedFromCache: false,
  chunks: [],
  correctionAttempts: 0,
  totalLatencyMs: 4231,
};

describe('AnswerPanel LangSmith trace link (A4)', () => {
  it('renders the trace link when langsmith_trace_url is present', () => {
    render(
      <AnswerPanel
        {...BASE_PROPS}
        langsmithTraceUrl="https://smith.langchain.com/o/org/projects/p/trace/abc123"
      />
    );

    const link = screen.getByTestId('langsmith-trace-link');
    expect(link).toHaveAttribute(
      'href',
      'https://smith.langchain.com/o/org/projects/p/trace/abc123'
    );
    // Opens in a new tab without leaking the opener context.
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
    expect(link).toHaveTextContent('TRACE');
  });

  it('renders no trace link when the URL is absent (tracing disabled)', () => {
    render(<AnswerPanel {...BASE_PROPS} />);

    expect(screen.queryByTestId('langsmith-trace-link')).not.toBeInTheDocument();
    // The rest of the answer panel still renders normally.
    expect(screen.getByText(/BM25 scores documents/)).toBeInTheDocument();
  });
});
