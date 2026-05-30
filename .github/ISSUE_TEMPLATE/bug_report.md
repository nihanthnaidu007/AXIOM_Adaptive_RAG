---
name: Bug report
about: Reproducible failure in AXIOM
title: "[Bug] "
labels: bug
---

## Summary

One sentence describing what went wrong.

## Reproduction steps

1.
2.
3.

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include error messages verbatim where possible.

## Environment

- AXIOM version (from header / status bar):
- Backend Python version (`python --version`):
- Node version (`node --version`):
- OS:
- Running with Docker Compose? (yes/no):
- `USE_CLAUDE_EVALUATOR` value:
- `TAVILY_API_KEY` configured? (yes/no — do not paste the key):

## Logs

Paste backend stderr around the failure (redact any keys):

```
```

If the failure is in the pipeline, attach the LangSmith trace URL if you have
one.

## Screenshots

If the UI is involved, add a screenshot showing the failure mode.

## Additional context

Anything else relevant: corpus size, ingested doc types, prior queries that
worked, etc.
