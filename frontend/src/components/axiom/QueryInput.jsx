import React from 'react';
import { Search, Zap } from 'lucide-react';
import { Button } from '../ui/button';

export const QueryInput = ({ 
  query, 
  setQuery, 
  onSubmit, 
  isLoading, 
  strategy 
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit();
    }
  };

  return (
    <div className="space-y-3" data-testid="query-input-section">
      <form onSubmit={handleSubmit}>
        <div className="axiom-query-wrap flex items-center">
          <div className="pl-4 text-violet-400">
            <Search size={18} />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question..."
            className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-[15px] text-white placeholder:text-gray-500"
            disabled={isLoading}
            data-testid="query-input"
          />
          <Button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="axiom-run-btn m-2"
            data-testid="run-query-btn"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <div className="loading-spinner w-4 h-4" />
                PROCESSING
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Zap size={14} />
                RUN
              </span>
            )}
          </Button>
        </div>
      </form>

      {/* Strategy Chips */}
      <div className="flex items-center gap-2 text-xs" data-testid="strategy-chips">
        <span className="text-gray-500 font-medium">Strategy auto-detected:</span>
        <span className={`strategy-chip bm25 ${strategy === 'bm25' ? 'active' : ''}`}>
          FACTUAL·BM25
        </span>
        <span className={`strategy-chip vector ${strategy === 'vector' ? 'active' : ''}`}>
          ABSTRACT·VECTOR
        </span>
        <span className={`strategy-chip hybrid ${strategy === 'hybrid' ? 'active' : ''}`}>
          HYBRID
        </span>
      </div>
    </div>
  );
};

export default QueryInput;
