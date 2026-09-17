import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, ArrowUp, ArrowDown, Minus } from 'lucide-react';
import { ScrollArea } from '../ui/scroll-area';

const ScorePips = ({ score, maxPips = 5 }) => {
  const filledPips = Math.round(score * maxPips);
  
  return (
    <div className="chunk-score-pips">
      {Array.from({ length: maxPips }).map((_, i) => (
        <div key={i} className={`pip ${i < filledPips ? 'filled' : ''}`} />
      ))}
    </div>
  );
};

const RerankDelta = ({ pre, post }) => {
  if (pre === null || post === null || pre === undefined || post === undefined) {
    return <span className="chunk-rerank-delta same">●</span>;
  }
  
  const delta = pre - post;
  
  if (delta > 0) {
    return (
      <span className="chunk-rerank-delta up flex items-center gap-1">
        <ArrowUp size={10} /> +{delta}
      </span>
    );
  } else if (delta < 0) {
    return (
      <span className="chunk-rerank-delta down flex items-center gap-1">
        <ArrowDown size={10} /> {delta}
      </span>
    );
  }
  
  return <span className="chunk-rerank-delta same"><Minus size={10} /></span>;
};

const ChunkCard = ({ chunk, index, expanded, onToggle }) => {
  const score = chunk.rerank_score || chunk.rrf_score || chunk.vector_score || chunk.bm25_score || 0;
  
  // Determine score type label
  let scoreLabel = 'Score';
  if (chunk.bm25_score !== null && chunk.bm25_score !== undefined) scoreLabel = 'BM25';
  if (chunk.vector_score !== null && chunk.vector_score !== undefined) scoreLabel = 'Similarity';
  if (chunk.rrf_score !== null && chunk.rrf_score !== undefined) scoreLabel = 'RRF';
  if (chunk.rerank_score !== null && chunk.rerank_score !== undefined) scoreLabel = 'Rerank';

  return (
    <div className="chunk-card" data-testid={`chunk-card-${index}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-violet-300">#{index + 1}</span>
          <RerankDelta pre={chunk.pre_rerank_position} post={chunk.post_rerank_position} />
          <span className="font-mono text-xs text-gray-400">
            {scoreLabel}: {score.toFixed(3)}
          </span>
          <ScorePips score={score} />
        </div>
        <button
          onClick={onToggle}
          className="text-gray-500 hover:text-violet-400 transition-colors"
          data-testid={`chunk-expand-${index}`}
        >
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>
      
      <div className="flex items-center gap-2 mb-2">
        <FileText size={12} className="text-gray-500" />
        <span className="text-xs text-gray-400">{chunk.source}</span>
      </div>
      
      <div className={`text-sm text-gray-300 ${expanded ? '' : 'line-clamp-2'}`}>
        {expanded ? chunk.content : chunk.content.substring(0, 180) + (chunk.content.length > 180 ? '...' : '')}
      </div>
    </div>
  );
};

export const SignalPanel = ({ chunks, strategy, parallelTiming, result }) => {
  const [expandedChunks, setExpandedChunks] = useState({});
  
  const toggleChunk = (index) => {
    setExpandedChunks(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="h-full flex flex-col" data-testid="signal-panel">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display text-sm font-semibold text-gray-300 tracking-wide">
          RETRIEVAL SIGNAL
        </h3>
        {strategy && (
          <span className={`strategy-chip ${strategy} active text-[10px]`}>
            {strategy.toUpperCase()}
          </span>
        )}
      </div>
      
      <div className="text-xs text-gray-500 mb-3 space-y-1">
        <div className="flex justify-between">
          <span>Chunks retrieved:</span>
          <span className="font-mono">{chunks?.length || 0}</span>
        </div>
        {parallelTiming && parallelTiming.speedup_factor && (
          <div className="flex justify-between">
            <span>Parallel speedup:</span>
            <span className="font-mono text-emerald-400">{parallelTiming.speedup_factor}x</span>
          </div>
        )}
      </div>
      
      {result?.decomposed && (
        <div className="decomposition-badge">
          <span>◈ MULTI-HOP</span>
          <span>{result.sub_query_results?.length} sub-queries executed in parallel</span>
          <div className="sub-queries-list">
            {result.classification?.sub_queries?.map((sq, i) => (
              <div key={i} className="sub-query-item">
                <span className="sub-query-num">Q{i + 1}</span>
                <span className="sub-query-text">{sq}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <ScrollArea className="flex-1">
        <div className="space-y-2 pr-2">
          {chunks && chunks.length > 0 ? (
            chunks.map((chunk, index) => (
              <ChunkCard
                key={chunk.chunk_id || index}
                chunk={chunk}
                index={index}
                expanded={expandedChunks[index]}
                onToggle={() => toggleChunk(index)}
              />
            ))
          ) : (
            <div className="text-center text-gray-500 py-8">
              <FileText size={24} className="mx-auto mb-2 opacity-30" />
              <p className="text-xs">No chunks retrieved yet</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default SignalPanel;
