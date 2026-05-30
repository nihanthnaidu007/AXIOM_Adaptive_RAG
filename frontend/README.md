# AXIOM Frontend

React 18 + Tailwind dashboard for the AXIOM pipeline. Consumes
`POST /api/query/stream` as Server-Sent Events and animates the 13-node
LangGraph trace one step at a time.

See the [root README](../README.md) for the full architecture, environment
setup, and pipeline documentation. This file only covers running the
frontend in isolation.

## Develop

```bash
npm install
npm start
```

Dev server runs on `http://localhost:3000` and proxies to a backend at
`http://localhost:8000` by default. Override with `REACT_APP_BACKEND_URL`.

## Build

```bash
npm run build
```

CRA emits an optimized bundle into `build/`. Source maps are intentionally
disabled — CI fails if any `.map` files are produced.

## Layout

```
src/
├── App.js                          Dashboard shell, SSE consumer, state wiring
├── index.css                       MERIDIAN design tokens + global styles
├── config.js                       Backend URL resolution
└── components/axiom/
    ├── QueryInput.js               Query bar + strategy auto-detect chips
    ├── UploadPanel.js              PDF / TXT / MD ingestion
    ├── PipelineStrip.js            13-node Signal Trace, hides web_search slot when inactive
    ├── SignalPanel.js              Reranked chunks with score + delta
    ├── EvaluationPanel.js          RAGAS metrics + per-iteration history
    ├── CorrectionRecord.js         Rewrite reasoning cards
    ├── AnswerPanel.js              Final answer + confidence band + sources
    └── StatusBar.js                Health pills, version, session stats
```

## Tests

There is no frontend unit-test suite yet. CI verifies the build succeeds and
no source maps leak. Manual smoke testing covers UI flows.
