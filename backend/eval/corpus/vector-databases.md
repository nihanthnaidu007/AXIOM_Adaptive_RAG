# Vector Databases at Scale

A vector database indexes embeddings for approximate nearest neighbor search
at scale. Vector indexes such as IVFFlat or HNSW partition the space so
queries touch only a subset of vectors, keeping latency low as the corpus
grows.

Scaling a vector database involves sharding, replication, memory budgeting,
and choosing the recall/speed tradeoff of the approximate index.

At production scale the index structure interacts with chunk retrieval:
smaller chunks give finer-grained retrieval but multiply the number of
vectors to index.
