# Production Upgrade — Delivery Checkpoint

This folder is the checkpoint paper trail of the four-repo production upgrade program (Nixus-Sql, AXIOM_Adaptive_RAG, Research_Forge, SpectraVoice), as applied to **AXIOM_Adaptive_RAG**. The program took AXIOM from a self-hosted document-QA prototype to a production-ready platform across phased waves: readiness survey → grounded spec → CI-governed delivery PRs.

The annotated tag `production-upgrade/final` marks AXIOM's final delivered state on `main` after Wave 2.

## Contents

| File | What it is | Source artifact |
|---|---|---|
| [dossier.md](dossier.md) | Final delivery dossier covering all four repos and Waves 0–2 — the program's final review handoff | `art_jV2n9Tta` |
| [master-plan.md](master-plan.md) | Cross-repo master plan: roadmap, acceptance criteria, CI-gated PR model | `art_gwXo54Vz` |
| [survey.md](survey.md) | AXIOM production-readiness survey — the program's starting point for this repo | `art_mEKQ5bhY` |
| [w1-spec.md](w1-spec.md) | AXIOM Wave 1 grounded spec — tests, CI, schema, error envelopes | `art_0TKxEV4g` |
| [w2-preview-review.md](w2-preview-review.md) | AXIOM Wave 2 PR #4 preview review findings | `art_foQIGjIf` |
| [w2-pr-record.md](w2-pr-record.md) | AXIOM Wave 2 PR #4 record with merge and CI evidence | `art_Fw8ix4e8` |

## Checkpoint

- Wave 2 chain on `main`: `da94af5` (#2) → `b4fe841` (#3) → `0878bfa` (#4)
- Final merged state: `0878bfa` — PR #4 CI 6/6 green
- Tag: `production-upgrade/final`
