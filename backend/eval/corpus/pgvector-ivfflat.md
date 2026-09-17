# IVFFlat Indexes in pgvector

pgvector is a PostgreSQL extension for vector similarity search. IVFFlat
stands for inverted file with flat storage: an IVFFlat index partitions
vectors into lists using k-means clustering and searches only the nearest
lists at query time.

This makes IVFFlat an approximate nearest neighbor index — it trades a
small amount of recall for speed. In pgvector you create an IVFFlat index
with a lists parameter and tune probes at query time. The flat component
stores raw vectors so distance computations are exact within the probed
lists, giving reliable similarity search for retrieval pipelines.
