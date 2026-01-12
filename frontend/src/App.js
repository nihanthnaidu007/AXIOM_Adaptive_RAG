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
import HexBackground from './components/axiom/HexBackground';
import QueryInput from './components/axiom/QueryInput';
import PipelineStrip from './components/axiom/PipelineStrip';
import SignalPanel from './components/axiom/SignalPanel';
import EvaluationPanel from './components/axiom/EvaluationPanel';
import CorrectionRecord from './components/axiom/CorrectionRecord';
import AnswerPanel from './components/axiom/AnswerPanel';
import StatusBar from './components/axiom/StatusBar';
import UploadPanel from './components/axiom/UploadPanel';

import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8000';
const API = `${BACKEND_URL}/api`;

// Main AXIOM Dashboard
const AxiomDashboard = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);

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
          const svc = response.data.services || {};
          const live =
            svc.postgres === 'connected' &&
            svc.redis === 'connected' &&
            svc.ollama === 'connected';
          const modeLabel = response.data.stub_mode
            ? 'Stub mode'
            : live
              ? 'Postgres · Redis · Ollama'
              : 'Live pipeline';
          toast.success('AXIOM Connected', {
            description: `${response.data.nodes.length} nodes • ${modeLabel}`,
          });
        }
      } catch (error) {
        toast.error('Connection Failed', {
          description: 'Unable to connect to AXIOM backend'
        });
      }
    };
    checkHealth();
  }, []);

  // Trigger hex background zone based on pipeline stage
  useEffect(() => {
    if (!result?.trace_steps) return;
    
    const lastStep = result.trace_steps[result.trace_steps.length - 1];
    if (!lastStep) return;

    const nodeName = lastStep.node_name;
    
    // Map nodes to zones
    if (['retrieve_bm25', 'retrieve_vector', 'retrieve_hybrid', 'rerank_chunks'].includes(nodeName)) {
      window.axiomSetZone?.('retrieval', 0.3);
    } else if (['generate_answer'].includes(nodeName)) {
      window.axiomSetZone?.('generation', 0.3);
    } else if (['evaluate_answer', 'finalize_answer'].includes(nodeName)) {
      window.axiomSetZone?.('evaluation', 0.3);
    } else {
      window.axiomSetZone?.(null, 0);
    }

    // Trigger hallucination pulse if detected
    if (result.hallucination_detected && lastStep.node_name === 'evaluate_answer') {
      window.axiomHallucinationPulse?.();
    }
  }, [result?.trace_steps, result?.hallucination_detected]);

  const handleSubmit = useCallback(async () => {
    if (!query.trim()) return;
    
    setIsLoading(true);
    setResult(null);
    
    try {
      const response = await axios.post(`${API}/query`, {
        query: query.trim(),
        session_id: null // Let backend generate
      });
      
      setResult(response.data);
      
      // Show success toast with confidence
      const confidence = response.data.confidence;
      if (confidence) {
        const faith = response.data.ragas_scores?.faithfulness;
        const faithStr =
          typeof faith === 'number' && !Number.isNaN(faith)
            ? faith.toFixed(2)
            : 'n/a';
        const cacheNote = response.data.served_from_cache ? ' (cache)' : '';
        toast.success(`Answer Generated: ${confidence.label}`, {
          description: `Faithfulness: ${faithStr}${cacheNote} • ${response.data.correction_attempts ?? 0} corrections`,
        });
      }
      
      // Update stats
      const statsResponse = await axios.get(`${API}/stats`);
      setStats(statsResponse.data);
      
    } catch (error) {
      console.error('Query failed:', error);
      toast.error('Query Failed', {
        description: error.response?.data?.detail?.error || error.message
      });
    } finally {
      setIsLoading(false);
      window.axiomSetZone?.(null, 0);
    }
  }, [query]);

  // Extract data from result
  const strategy = result?.retrieval_strategy || result?.classification?.retrieval_strategy || '';
  const chunks = result?.reranked_chunks || [];
  const scoresHistory = result?.scores_history || [];
  const corrections = result?.correction_history || [];
  const traceSteps = result?.trace_steps || [];

  return (
    <div className="min-h-screen flex flex-col relative" data-testid="axiom-dashboard">
      {/* Hex Background */}
      <HexBackground />
      
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
          <span className="font-mono">v1.0</span>
          <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-violet-400 animate-pulse' : 'bg-emerald-400'}`} />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 relative z-10 flex flex-col gap-6">
        {/* Query Input */}
        <QueryInput
          query={query}
          setQuery={setQuery}
          onSubmit={handleSubmit}
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
        />
      </main>

      {/* Status Bar */}
      <StatusBar stats={stats} isProcessing={isLoading} result={result} />
      
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
