import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, Check, Loader2 } from 'lucide-react';
import { fetchJson } from '../../lib/api';

/**
 * Thumbs feedback widget (Wave 2, D2).
 *
 * Rate the current answer ±1 with optional free text; POSTs to
 * /api/feedback keyed by the session trace id. Persistence only — the
 * retrieval-tuning loop consuming this data is a later wave. Renders
 * nothing until there is a trace to attach feedback to.
 */
export default function FeedbackWidget({ traceId, className = '' }) {
  const [rating, setRating] = useState(null);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  if (!traceId) return null;

  const submit = async (chosenRating) => {
    setSubmitting(true);
    setError(null);
    try {
      await fetchJson('/feedback', {
        method: 'POST',
        body: JSON.stringify({
          trace_id: traceId,
          rating: chosenRating,
          comment: comment.trim() ? comment.trim() : null,
        }),
      });
      setSubmitted(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div
        className="flex items-center gap-2 text-xs text-emerald-400"
        data-testid="feedback-submitted"
      >
        <Check size={13} />
        Feedback recorded — thank you.
      </div>
    );
  }

  return (
    <div
      className={`rounded-lg border border-violet-500/10 bg-[hsl(var(--bg-panel)/0.5)] p-4 ${className}`}
      data-testid="feedback-widget"
    >
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-400">Was this answer helpful?</span>
        <button
          type="button"
          onClick={() => setRating(1)}
          disabled={submitting}
          aria-label="Thumbs up"
          aria-pressed={rating === 1}
          className={`p-1.5 rounded transition-colors ${
            rating === 1
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'text-gray-400 hover:text-gray-200 hover:bg-violet-500/10'
          } disabled:opacity-50`}
        >
          <ThumbsUp size={14} />
        </button>
        <button
          type="button"
          onClick={() => setRating(-1)}
          disabled={submitting}
          aria-label="Thumbs down"
          aria-pressed={rating === -1}
          className={`p-1.5 rounded transition-colors ${
            rating === -1
              ? 'bg-red-500/20 text-red-300'
              : 'text-gray-400 hover:text-gray-200 hover:bg-violet-500/10'
          } disabled:opacity-50`}
        >
          <ThumbsDown size={14} />
        </button>
        {submitting && (
          <Loader2 size={13} className="animate-spin text-violet-400" data-testid="feedback-loading" />
        )}
      </div>

      {/* The form stays visible on error so the user can retry directly */}
      {rating !== null && (
        <div className="mt-3 flex flex-col gap-2" data-testid="feedback-form">
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Optional — what was right or wrong?"
            rows={2}
            data-testid="feedback-comment"
            className="w-full rounded bg-black/30 border border-violet-500/10 p-2 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-violet-500/40"
          />
          <button
            type="button"
            onClick={() => submit(rating)}
            disabled={submitting}
            data-testid="feedback-submit"
            className="self-start rounded bg-violet-600/30 border border-violet-500/30 px-3 py-1 text-xs text-violet-200 hover:bg-violet-600/50 transition-colors disabled:opacity-50"
          >
            Submit feedback
          </button>
        </div>
      )}

      {error && (
        <p className="mt-2 text-xs text-red-400" data-testid="feedback-error">
          {error}
        </p>
      )}
    </div>
  );
}
