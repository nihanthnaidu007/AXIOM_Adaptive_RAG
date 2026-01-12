import React from 'react';
import { RotateCcw, ArrowRight } from 'lucide-react';

export const CorrectionRecord = ({ corrections }) => {
  if (!corrections || corrections.length === 0) return null;

  return (
    <div className="space-y-3" data-testid="correction-record">
      <div className="section-header">
        Correction Record
        <span className="text-amber-400 font-mono text-[10px] ml-2">
          [{corrections.length} iteration{corrections.length > 1 ? 's' : ''}]
        </span>
      </div>

      {corrections.map((record, idx) => (
        <div key={idx} className="correction-record">
          <div className="correction-record-header flex items-center gap-2">
            <RotateCcw size={12} />
            ITERATION {record.iteration}
          </div>
          
          <div className="correction-reasoning">
            "{record.rewrite_reasoning}"
          </div>
          
          <div className="mt-3 flex items-start gap-2">
            <ArrowRight size={14} className="text-violet-400 mt-0.5 flex-shrink-0" />
            <div className="rewritten-query">
              {record.rewritten_query}
            </div>
          </div>
          
          <div className="mt-3 text-[10px] text-gray-500 flex gap-4">
            <span>
              Previous faithfulness: 
              <span className="text-red-400 font-mono ml-1">
                {record.ragas_scores_before?.faithfulness?.toFixed(2)}
              </span>
            </span>
            <span>
              Strategy: 
              <span className="text-violet-300 font-mono ml-1">
                {record.retrieval_strategy_used?.toUpperCase()}
              </span>
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default CorrectionRecord;
