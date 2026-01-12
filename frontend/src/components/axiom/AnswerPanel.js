import React from 'react';
import { Shield, ShieldCheck, ShieldAlert, ShieldX, Clock, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

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

export const AnswerPanel = ({ 
  answer, 
  confidence, 
  isLoading, 
  servedFromCache,
  chunks,
  correctionAttempts,
  totalLatencyMs
}) => {
  const totalLatency = totalLatencyMs != null ? Math.round(totalLatencyMs) : 0;

  return (
    <div className="answer-panel" data-testid="answer-panel">
      <div className="answer-panel-header">
        <ConfidenceBadge confidence={confidence} />
        
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
        {isLoading ? (
          <div className="flex items-center gap-2 text-gray-400">
            <div className="loading-spinner" />
            <span>Generating answer...</span>
          </div>
        ) : answer ? (
          <>
            <ReactMarkdown>{answer}</ReactMarkdown>
            {/* Source attribution */}
            {chunks && chunks.length > 0 && (
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
          {confidence.reasoning}
        </div>
      )}
    </div>
  );
};

export default AnswerPanel;
