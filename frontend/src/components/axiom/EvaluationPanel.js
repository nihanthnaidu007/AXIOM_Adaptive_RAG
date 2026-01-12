import React from 'react';
import { AlertTriangle, CheckCircle, ArrowUp, ArrowDown } from 'lucide-react';

const FAITHFULNESS_THRESHOLD = 0.75;
const RELEVANCY_THRESHOLD = 0.70;
const GROUNDEDNESS_THRESHOLD = 0.65;

const ScoreBar = ({ label, value, threshold, isKey = false }) => {
  if (value == null) {
    return (
      <div className="score-row">
        <span className={`score-label ${isKey ? 'font-semibold' : ''}`}>{label}</span>
        <div className="score-bar-track" />
        <span className="score-value" style={{ color: '#6b7280' }}>N/A</span>
      </div>
    );
  }
  const percentage = value * 100;
  const passed = value >= threshold;
  const nearThreshold = value >= threshold - 0.15 && value < threshold;
  
  const colorClass = passed ? 'pass' : nearThreshold ? 'warn' : 'fail';
  
  return (
    <div className={`score-row ${isKey ? 'bg-opacity-5' : ''}`} style={isKey && !passed ? { background: 'rgba(220, 38, 38, 0.06)' } : isKey && passed ? { background: 'rgba(5, 150, 105, 0.04)' } : {}}>
      <span className={`score-label ${isKey ? 'font-semibold' : ''}`}>
        {label}
        {isKey && (
          <span className="ml-2 text-[10px] text-gray-500">◆ GATE</span>
        )}
      </span>
      <div className="score-bar-track">
        <div 
          className={`score-bar-fill ${colorClass}`}
          style={{ width: `${percentage}%` }}
        />
        <div 
          className="threshold-marker"
          style={{ left: `${threshold * 100}%` }}
          title={`Threshold: ${threshold}`}
        />
      </div>
      <span className={`score-value ${colorClass}`}>
        {value.toFixed(2)}
        {passed ? (
          <CheckCircle size={10} className="inline ml-1" />
        ) : (
          <AlertTriangle size={10} className="inline ml-1" />
        )}
      </span>
    </div>
  );
};

const ScoreDelta = ({ current, previous, label }) => {
  if (!previous || current == null || previous == null) return null;

  const delta = current - previous;
  const isImprovement = delta > 0;
  
  return (
    <span className={`text-[10px] ml-2 ${isImprovement ? 'text-emerald-400' : 'text-red-400'}`}>
      {isImprovement ? <ArrowUp size={10} className="inline" /> : <ArrowDown size={10} className="inline" />}
      {delta > 0 ? '+' : ''}{delta.toFixed(2)}
    </span>
  );
};

export const EvaluationPanel = ({ scoresHistory, hallucinationDetected, evaluationPassed }) => {
  const currentScores = scoresHistory?.[scoresHistory.length - 1];
  const previousScores = scoresHistory?.length > 1 ? scoresHistory[scoresHistory.length - 2] : null;
  const iteration = scoresHistory?.length || 0;

  return (
    <div className="h-full flex flex-col" data-testid="evaluation-panel">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display text-sm font-semibold text-gray-300 tracking-wide">
          EVALUATION SIGNAL
        </h3>
        {iteration > 0 && (
          <span className="text-xs text-gray-500 font-mono">
            Iteration {iteration}
          </span>
        )}
      </div>

      {currentScores ? (
        <div className="space-y-4">
          {/* RAGAS Scores */}
          <div className="space-y-1">
            <ScoreBar
              label="Faithfulness"
              value={currentScores.faithfulness}
              threshold={FAITHFULNESS_THRESHOLD}
              isKey={true}
            />
            <ScoreBar
              label="Relevancy"
              value={currentScores.answer_relevancy}
              threshold={RELEVANCY_THRESHOLD}
            />
            <ScoreBar
              label="Groundedness"
              value={currentScores.context_groundedness}
              threshold={GROUNDEDNESS_THRESHOLD}
            />
          </div>

          {/* Composite Score */}
          <div className="flex items-center justify-between text-xs py-2 border-t border-violet-500/10">
            <span className="text-gray-400">Composite Score:</span>
            <span className="font-mono text-violet-300">
              {currentScores.composite_score?.toFixed(3)}
              {previousScores && (
                <ScoreDelta 
                  current={currentScores.composite_score} 
                  previous={previousScores.composite_score}
                />
              )}
            </span>
          </div>

          {/* Status Badge */}
          {hallucinationDetected && !evaluationPassed ? (
            <div className="hallucination-badge">
              <AlertTriangle size={12} />
              HALLUCINATION DETECTED
            </div>
          ) : evaluationPassed ? (
            <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold">
              <CheckCircle size={14} />
              EVALUATION PASSED
            </div>
          ) : null}

          {/* Score History Summary */}
          {scoresHistory && scoresHistory.length > 1 && (
            <div className="mt-4 pt-4 border-t border-violet-500/10">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">
                Score History
              </div>
              <div className="flex gap-2">
                {scoresHistory.map((score, idx) => (
                  <div 
                    key={idx}
                    className={`px-2 py-1 rounded text-[10px] font-mono ${
                      (score.faithfulness ?? 0) >= FAITHFULNESS_THRESHOLD
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-red-500/10 text-red-400'
                    }`}
                  >
                    #{idx + 1}: {score.faithfulness != null ? score.faithfulness.toFixed(2) : 'N/A'}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scorer Model */}
          <div className="text-[10px] text-gray-600 mt-2">
            Scorer: {currentScores.scorer_model || 'mock'}
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500 py-8">
          <AlertTriangle size={24} className="mx-auto mb-2 opacity-30" />
          <p className="text-xs">Awaiting evaluation</p>
        </div>
      )}
    </div>
  );
};

export default EvaluationPanel;
