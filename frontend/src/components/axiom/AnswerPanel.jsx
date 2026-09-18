import React from 'react';
import { Shield, ShieldCheck, ShieldAlert, ShieldX, Clock, Zap, Globe, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const WebSourcedBadge = ({ webChunkCount }) => (
  <span className="flex items-center gap-1.5 text-[10px] font-mono text-sky-400/70 px-2 py-0.5 rounded border border-sky-500/20 bg-sky-500/5">
    <Globe size={9} />
    WEB · {webChunkCount} result{webChunkCount !== 1 ? 's' : ''}
  </span>
);

const ConfidenceBadge = ({ confidence }) => {
  if (!confidence) return null;

  const { label, score } = confidence;
  const labelLower = label.toLowerCase();

  const icons = {
    verified: ShieldCheck,
    probable: Shield,
    uncertain: ShieldAlert,
    unreliable: ShieldX
  };

  const Icon = icons[labelLower] || Shield;

  return (
    <div className="flex items-center gap-3">
      <span className={`confidence-label ${labelLower}`}>
        <Icon size={12} className="inline mr-1" />
        {label}
      </span>
      <span className="font-mono text-xs text-gray-400">
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  );
};

const ConfidenceTrack = ({ confidence }) => {
  if (!confidence) return null;

  const { label, score } = confidence;
  const labelLower = label.toLowerCase();

  const colorClasses = {
    verified: 'bg-emerald-500',
    probable: 'bg-cyan-500',
    uncertain: 'bg-amber-500',
    unreliable: 'bg-red-500'
  };

  return (
    <div className="confidence-track">
      <div 
        className={`confidence-track-fill ${colorClasses[labelLower] || 'bg-violet-500'}`}
        style={{ width: `${score * 100}%` }}
      />
    </div>
  );
};

/**
 * Stage line shown while the answer is streaming in (retrieving → generating).
 */
const StreamStageIndicator = ({ streamStage }) => {
  const label = streamStage === 'retrieving'
    ? 'Retrieving context...'
    : 'Generating answer...';

  return (
    <div className="flex items-center gap-2 text-gray-400" data-testid="stream-stage">
      <div className="loading-spinner" />
      <span>{label}</span>
    </div>
  );
};

/**
 * Citations from the SSE "sources" event: document chunks (source, score,
 * preview) and web results (linked). Rendered as soon as the event arrives,
 * before the final result lands.
 */
const StreamCitations = ({ streamSources }) => {
  if (!streamSources || !streamSources.sources?.length) return null;

  const docs = streamSources.sources.filter((s) => s.kind === 'document');
  const web = streamSources.sources.filter((s) => s.kind === 'web');

  return (
    <div className="mt-4 pt-4 border-t border-violet-500/10 text-xs text-gray-500" data-testid="stream-citations">
      <span className="text-gray-400">Citations: </span>
      {docs.slice(0, 5).map((doc, idx) => (
        <span key={`doc-${idx}`} title={doc.preview || ''}>
          {idx > 0 && ' · '}
          <span className="text-violet-400">{doc.source || 'document'}</span>
          {doc.score != null && (
            <span className="text-gray-600 ml-1">({Number(doc.score).toFixed(2)})</span>
          )}
        </span>
      ))}
      {web.map((src, idx) => {
        let hostname = src.url || 'web';
        try { hostname = new URL(src.url).hostname.replace('www.', ''); } catch { /* hostname falls back to the raw URL */ }
        return (
          <span key={`web-${idx}`}>
            {(docs.length > 0 || idx > 0) && ' · '}
            <a
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-400/80 hover:text-sky-300 transition-colors"
              title={src.title || src.url}
            >
              {src.title ? src.title.slice(0, 40) + (src.title.length > 40 ? '…' : '') : hostname}
            </a>
          </span>
        );
      })}
    </div>
  );
};

export const AnswerPanel = ({
  answer,
  confidence,
  isLoading,
  isStreaming = false,
  streamStage = null,
  streamSources = null,
  servedFromCache,
  chunks,
  correctionAttempts,
  totalLatencyMs,
  documentChunkCount = 0,
  webSearchUsed = false,
  webChunkCount = 0,
  webSearchChunks = [],
  langsmithTraceUrl = null,
}) => {
  const totalLatency = totalLatencyMs != null ? Math.round(totalLatencyMs) : 0;

  return (
    <div className="answer-panel" data-testid="answer-panel">
      <div className="answer-panel-header">
        <div className="flex items-center gap-2">
          <ConfidenceBadge confidence={confidence} />
          {webSearchUsed && webChunkCount > 0 && (
            <WebSourcedBadge webChunkCount={webChunkCount} />
          )}
          {webSearchUsed && webChunkCount === 0 && !isLoading && (
            <span className="flex items-center gap-1.5 text-[10px] font-mono text-gray-500 px-2 py-0.5 rounded border border-gray-700/40">
              <Globe size={9} />
              WEB · no results
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {servedFromCache && (
            <span className="cache-hit-badge">
              <Zap size={10} className="inline mr-1" />
              CACHE HIT
            </span>
          )}
          {correctionAttempts > 0 && (
            <span className="text-[10px] text-amber-400 font-mono">
              {correctionAttempts} correction{correctionAttempts > 1 ? 's' : ''}
            </span>
          )}
          {/* LangSmith trace link — only present when tracing is enabled
              server-side; the field is null otherwise, so no dead link. */}
          {langsmithTraceUrl && (
            <a
              href={langsmithTraceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] text-violet-400/80 hover:text-violet-300 font-mono flex items-center gap-1 transition-colors"
              title="Open the full LangSmith trace for this query"
              data-testid="langsmith-trace-link"
            >
              <ExternalLink size={10} />
              TRACE
            </a>
          )}
          {totalLatency > 0 && (
            <span className="text-[10px] text-gray-500 font-mono flex items-center gap-1">
              <Clock size={10} />
              {totalLatency}ms
            </span>
          )}
        </div>
      </div>

      <ConfidenceTrack confidence={confidence} />

      <div className="answer-text">
        {isLoading && !answer ? (
          <StreamStageIndicator streamStage={streamStage} />
        ) : answer ? (
          <>
            <ReactMarkdown>{answer}</ReactMarkdown>
            {isStreaming && (
              <span className="streaming-cursor" aria-hidden="true">▍</span>
            )}
            {/* Citations from the streaming "sources" event */}
            <StreamCitations streamSources={streamSources} />
            {/* Source attribution */}
            {chunks && chunks.length > 0 && documentChunkCount > 0 && (
              <div className="mt-4 pt-4 border-t border-violet-500/10 text-xs text-gray-500">
                <span className="text-gray-400">Sources: </span>
                {chunks.slice(0, 3).map((chunk, idx) => (
                  <span key={idx}>
                    {idx > 0 && ' · '}
                    <span className="text-violet-400">
                      {chunk.source}
                    </span>
                    <span className="text-gray-600 ml-1">
                      ({chunk.rerank_score?.toFixed(2) || chunk.rrf_score?.toFixed(2) || '—'})
                    </span>
                  </span>
                ))}
              </div>
            )}
            {/* Web source attribution — shown when web search returned results */}
            {webSearchUsed && webChunkCount > 0 && webSearchChunks.length > 0 && documentChunkCount === 0 && (
              <div className="mt-4 pt-4 border-t border-violet-500/10 text-xs text-gray-500">
                <span className="text-gray-400">Web sources: </span>
                {webSearchChunks.slice(0, 3).map((chunk, idx) => {
                  let hostname = chunk.url;
                  try { hostname = new URL(chunk.url).hostname.replace('www.', ''); } catch { /* hostname falls back to the raw URL */ }
                  return (
                    <span key={idx}>
                      {idx > 0 && ' · '}
                      <a
                        href={chunk.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sky-400/80 hover:text-sky-300 transition-colors"
                        title={chunk.title || chunk.url}
                      >
                        {chunk.title ? chunk.title.slice(0, 40) + (chunk.title.length > 40 ? '…' : '') : hostname}
                      </a>
                    </span>
                  );
                })}
              </div>
            )}
          </>
        ) : (
          <span className="text-gray-500">
            Enter a query to get started
          </span>
        )}
      </div>

      {/* Confidence reasoning */}
      {confidence?.reasoning && !isLoading && (
        <div className="mt-4 text-[11px] text-gray-500 italic">
          {webSearchUsed && webChunkCount > 0
            ? `Web-sourced answer. ${
                confidence.label === 'UNRELIABLE'
                  ? 'Low confidence reflects web snippet quality — not a pipeline failure. Verify claims via the sources above.'
                  : 'Confidence reflects how well the web content grounded this answer.'
              }`
            : webSearchUsed && webChunkCount === 0
            ? 'Web search was triggered but returned no usable results. The corpus also had no matching documents.'
            : confidence.reasoning
          }
        </div>
      )}
    </div>
  );
};

export default AnswerPanel;
