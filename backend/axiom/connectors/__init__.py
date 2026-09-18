"""AXIOM ingestion connectors (Wave 4).

Connectors fetch documents from external sources (S3 buckets, websites) and
funnel them through the ONE shared ingestion path — parse_document →
DocumentChunker → indexer.index_run → per-document bookkeeping — with stable
source keys (S3 URI / URL) so re-runs replace idempotently. The RUN is the
batch boundary: one indexing pass and one semantic-cache clear per run.
"""
