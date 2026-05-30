import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { Toaster, toast } from 'sonner';
import { 
  ResizableHandle, 
  ResizablePanel, 
  ResizablePanelGroup 
} from './components/ui/resizable';

// AXIOM Components
import QueryInput from './components/axiom/QueryInput';
import PipelineStrip from './components/axiom/PipelineStrip';
import SignalPanel from './components/axiom/SignalPanel';
import EvaluationPanel from './components/axiom/EvaluationPanel';
import CorrectionRecord from './components/axiom/CorrectionRecord';
import AnswerPanel from './components/axiom/AnswerPanel';
import StatusBar from './components/axiom/StatusBar';
import UploadPanel from './components/axiom/UploadPanel';

import './App.css';
import { API_BASE_URL } from './config';

const API = `${API_BASE_URL}/api`;

/**
 * Apply a completed QueryResponse payload to all relevant React state setters.
 * Called once when the SSE "done" event arrives.
 */
function applyQueryResult(data, setters) {
  const { setResult, setTraceSteps, setSessionId } = setters;
  setResult(data);
  setTraceSteps(data.trace_steps || []);
  if (data.session_id) {
    setSessionId(data.session_id);
  }
}

// Main AXIOM Dashboard
const AxiomDashboard = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [traceSteps, setTraceSteps] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API}/health`);
        if (response.data.status === 'ok') {
          const sh = response.data.system_health || {};
          const stubMode = response.data.stub_mode;

          const evaluatorLabel = sh.evaluator
            ? sh.evaluator.replace('claude-haiku', 'Claude Haiku').replace('ollama', 'Ollama')
            : 'unknown';

          const webLabel = sh.web_search === 'tavily' ? ' · Tavily' : '';

          const modeLabel = stubMode
            ? 'Degraded — check /api/health'
            : `${evaluatorLabel}${webLabel}`;

          toast.success('AXIOM Connected', {
            description: `${(response.data.nodes || []).length} nodes · ${modeLabel}`,
          });
          setSystemHealth(response.data.system_health || null);
        }
      } catch (error) {
        toast.error('Connection Failed', {
          description: 'Unable to connect to AXIOM backend'
        });
      }
    };
    checkHealth();
  }, []);

  const handleSubmitStreaming = useCallback(async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setResult(null);
    setTraceSteps([]);

    try {
      const response = await fetch(`${API}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), session_id: sessionId || null }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamDone = false;

      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;

          const raw = line.slice(6).trim();
          if (raw === '[DONE]') {
            streamDone = true;
            break;
          }

          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }

          if (event.type === 'node_complete' && event.trace_step) {
            setTraceSteps(prev => [...prev, event.trace_step]);
          } else if (event.type === 'done' && event.result) {
            applyQueryResult(event.result, { setResult, setTraceSteps, setSessionId });

            const conf = event.result.confidence || {};
            const scores = event.result.ragas_scores || {};
            const faithStr = scores.faithfulness != null
              ? scores.faithfulness.toFixed(2)
              : 'n/a';
            const cacheNote = event.result.served_from_cache ? ' (cache)' : '';
            toast.success(`Answer Generated: ${conf.label || '—'}`, {
              description: `Faithfulness: ${faithStr}${cacheNote} · ${event.result.correction_attempts ?? 0} corrections`,
            });

            try {
              const statsResponse = await axios.get(`${API}/stats`);
              setStats(statsResponse.data);
            } catch {
              // Non-critical
            }
          } else if (event.type === 'error') {
            toast.error('Query Failed', {
              description: event.message || 'Unknown error from stream',
            });
          }
        }
      }
    } catch (error) {
      console.error('Streaming query failed:', error);
      toast.error('Query Failed', {
        description: error.message,
      });
    } finally {
      setIsLoading(false);
    }
  }, [query, sessionId, isLoading]);

  // Extract data from result
  const strategy = result?.retrieval_strategy || result?.classification?.retrieval_strategy || '';
  const chunks = result?.reranked_chunks || [];
  const scoresHistory = result?.scores_history || [];
  const corrections = result?.correction_history || [];

  return (
    <div className="min-h-screen flex flex-col relative" data-testid="axiom-dashboard">
      {/* Header */}
      <header className="axiom-header relative z-10">
        <div className="flex items-center gap-3">
          <span className="axiom-wordmark">
            <span>◆</span> AXIOM
          </span>
          <span className="text-xs text-gray-500 tracking-widest">
            ADAPTIVE INTELLIGENCE PLATFORM
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="font-mono">v1.5</span>
          <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-violet-400 animate-pulse' : 'bg-emerald-400'}`} />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 relative z-10 flex flex-col gap-6">
        {/* Query Input */}
        <QueryInput
          query={query}
          setQuery={setQuery}
          onSubmit={handleSubmitStreaming}
          isLoading={isLoading}
          strategy={strategy}
        />

        {/* Upload Panel */}
        <UploadPanel onDocsUpdated={fetchStats} />

        {/* Pipeline Strip */}
        <PipelineStrip
          traceSteps={traceSteps}
          correctionAttempts={result?.correction_attempts || 0}
          strategy={strategy}
          webSearchUsed={result?.web_search_used ?? false}
        />

        {/* Signal Panels - Resizable */}
        <div className="flex-1 min-h-[300px]">
          <ResizablePanelGroup direction="horizontal" className="rounded-lg border border-violet-500/10">
            <ResizablePanel defaultSize={50} minSize={30}>
              <div className="h-full p-4 bg-[hsl(var(--bg-panel)/0.5)]">
                <SignalPanel 
                  chunks={chunks}
                  strategy={strategy}
                  parallelTiming={result?.parallel_timing}
                  result={result}
                />
              </div>
            </ResizablePanel>
            
            <ResizableHandle withHandle className="bg-violet-500/10" />
            
            <ResizablePanel defaultSize={50} minSize={30}>
              <div className="h-full p-4 bg-[hsl(var(--bg-panel)/0.5)]">
                <EvaluationPanel
                  scoresHistory={scoresHistory}
                  hallucinationDetected={result?.hallucination_detected}
                  evaluationPassed={result?.evaluation_passed}
                  webSearchUsed={result?.web_search_used ?? false}
                />
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>

        {/* Correction Record (if any) */}
        {corrections.length > 0 && (
          <CorrectionRecord corrections={corrections} />
        )}

        {/* Answer Panel */}
        <AnswerPanel
          answer={result?.final_answer}
          confidence={result?.confidence}
          isLoading={isLoading}
          servedFromCache={result?.served_from_cache}
          chunks={chunks}
          correctionAttempts={result?.correction_attempts || 0}
          totalLatencyMs={result?.total_latency_ms}
          documentChunkCount={result?.document_chunk_count ?? 0}
          webSearchUsed={result?.web_search_used ?? false}
          webChunkCount={result?.web_chunk_count ?? 0}
          webSearchChunks={result?.web_search_chunks ?? []}
        />
      </main>

      {/* Status Bar */}
      <StatusBar stats={stats} isProcessing={isLoading} result={result} systemHealth={systemHealth} />
      
      {/* Toast Container */}
      <Toaster 
        position="bottom-right" 
        theme="dark"
        toastOptions={{
          style: {
            background: 'hsl(248 60% 5%)',
            border: '1px solid hsl(263 84% 58% / 0.25)',
            color: 'hsl(250 60% 98%)'
          }
        }}
      />
    </div>
  );
};

function App() {
  return (
    <div className="App dark">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AxiomDashboard />} />
          <Route path="*" element={<AxiomDashboard />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
