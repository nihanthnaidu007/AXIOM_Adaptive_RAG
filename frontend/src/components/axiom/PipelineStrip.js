import React from 'react';
import { CheckCircle, Circle, AlertCircle, RotateCcw, Clock } from 'lucide-react';

const PIPELINE_NODES = [
  { id: 'classify_query', label: 'classify' },
  { id: 'check_cache', label: 'cache' },
  { id: 'route_retrieval', label: 'route' },
  { id: 'retrieve_bm25', label: 'bm25', conditional: true },
  { id: 'retrieve_vector', label: 'vector', conditional: true },
  { id: 'retrieve_hybrid', label: 'hybrid', conditional: true },
  { id: 'decompose_query', label: 'decompose' },
  { id: 'rerank_chunks', label: 'rerank' },
  { id: 'generate_answer', label: 'generate' },
  { id: 'evaluate_answer', label: 'evaluate' },
  { id: 'rewrite_query', label: 'rewrite', loop: true },
  { id: 'finalize_answer', label: 'finalize' },
];

const getNodeStatus = (nodeName, traceSteps) => {
  if (!traceSteps || traceSteps.length === 0) return 'pending';
  
  const step = traceSteps.find(s => s.node_name === nodeName);
  if (!step) return 'pending';
  
  return step.status;
};

const NodeIcon = ({ status }) => {
  switch (status) {
    case 'complete':
      return <CheckCircle size={12} className="text-emerald-400" />;
    case 'running':
      return <Clock size={12} className="text-violet-400 animate-pulse" />;
    case 'error':
      return <AlertCircle size={12} className="text-red-400" />;
    case 'skipped':
      return <Circle size={12} className="text-gray-600" />;
    default:
      return <Circle size={12} className="text-gray-600" />;
  }
};

export const PipelineStrip = ({ traceSteps, correctionAttempts, strategy }) => {
  // Filter nodes based on strategy
  const visibleNodes = PIPELINE_NODES.filter(node => {
    if (!node.conditional) return true;
    if (node.id === 'retrieve_bm25') return strategy === 'bm25';
    if (node.id === 'retrieve_vector') return strategy === 'vector';
    if (node.id === 'retrieve_hybrid') return strategy === 'hybrid' || !strategy;
    return true;
  });

  return (
    <div className="space-y-2" data-testid="pipeline-strip">
      <div className="section-header">
        Signal Trace
      </div>
      
      <div className="pipeline-strip">
        {visibleNodes.map((node, index) => {
          const status = getNodeStatus(node.id, traceSteps);
          const isActive = index > 0 && 
            getNodeStatus(visibleNodes[index - 1].id, traceSteps) === 'complete' &&
            status !== 'pending';
          
          return (
            <React.Fragment key={node.id}>
              {index > 0 && (
                <div className={`pipeline-arrow ${isActive ? 'active' : ''}`} />
              )}
              <div 
                className={`pipeline-node ${status}`}
                data-testid={`pipeline-node-${node.id}`}
              >
                <NodeIcon status={status} />
                <span className="mt-1">{node.label}</span>
              </div>
            </React.Fragment>
          );
        })}
        
        {/* Correction Loop Indicator */}
        {correctionAttempts > 0 && (
          <div className="ml-4 flex items-center gap-2">
            <RotateCcw size={12} className="text-amber-400" />
            <span className="correction-loop-badge">
              ↺ {correctionAttempts}/3
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PipelineStrip;
