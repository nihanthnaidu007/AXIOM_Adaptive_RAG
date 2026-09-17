"""AXIOM Configuration - All environment variables and settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

from axiom.provider_mode import (
    LOCAL_PROVIDER,
    PROVIDERS,
    resolve_embedding_dimensions,
)


class AxiomConfig(BaseSettings):
    """AXIOM configuration loaded from environment variables."""

    # LLM
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "axiom"
    postgres_password: str = ""
    postgres_db: str = "axiom_rag"
    postgres_url: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # API authentication
    api_key: str = ""

    # Retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5

    # Evaluation thresholds
    faithfulness_threshold: float = 0.75
    relevancy_threshold: float = 0.70
    groundedness_threshold: float = 0.65
    cache_similarity_threshold: float = 0.95

    # Pipeline
    max_correction_attempts: int = 3

    # Run `alembic upgrade head` on startup before any store touches the DB.
    # Set RUN_MIGRATIONS_ON_STARTUP=false to manage migrations out-of-band.
    run_migrations_on_startup: bool = True

    # API limits
    max_query_length: int = 2000
    max_ingest_size_mb: int = 50
    rate_limit_per_minute: int = 30

    # Ollama (local critic)
    ollama_host: str = "http://localhost:11434"
    ollama_critic_model: str = "llama3.2"

    # Provider modes — "cloud" (default; unchanged for existing deployments)
    # or "local" (Ollama). Fully-local mode sets BOTH to local: a local
    # generator with cloud embeddings would keep a hard cloud dependency (and
    # a required OPENAI_API_KEY) inside a mode claiming local operation.
    llm_provider: str = "cloud"
    # Distinct from the critic default so critic != generator out of the box —
    # same-model self-grading correlates errors. Equal settings are allowed
    # but carry that documented caveat.
    ollama_generation_model: str = "llama3.1:8b"
    embedding_provider: str = "cloud"
    ollama_embedding_model: str = "nomic-embed-text"
    # Native width of the local embedding model. Owned in code: Ollama's
    # OpenAI-compat layer does not honor OpenAI's `dimensions` kwarg.
    local_embedding_dimensions: int = 768

    # Evaluation backend
    use_claude_evaluator: bool = True  # False to use local Ollama instead

    # Web Search (Tavily)
    tavily_api_key: str = ""
    # Search depth: "advanced" costs ~2x credits but returns full content.
    # "basic" returns short snippets (~300 chars). Set per-call based on corpus state.
    # Config default is "basic". The web_search_node overrides to "advanced"
    # automatically when the document corpus is empty.
    tavily_search_depth: str = "basic"
    # Maximum number of web results to fetch per search call.
    tavily_max_results: int = 5

    # LangSmith observability
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "axiom-rag"

    model_config = {
        # Load the project-root .env regardless of process working directory.
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def effective_embedding_model(self) -> str:
        """The embedding model the active provider mode actually calls."""
        if self.embedding_provider == LOCAL_PROVIDER:
            return self.ollama_embedding_model
        return self.embedding_model

    @property
    def effective_embedding_dimensions(self) -> int:
        """Vector width the active provider mode produces and stores."""
        return resolve_embedding_dimensions(
            self.embedding_provider, self.embedding_dimensions, self.local_embedding_dimensions
        )

    @model_validator(mode="after")
    def validate_required_keys(self) -> "AxiomConfig":
        for field in ("llm_provider", "embedding_provider"):
            provider = getattr(self, field)
            if provider not in PROVIDERS:
                raise ValueError(
                    f"{field.upper()} must be one of {', '.join(PROVIDERS)} (got {provider!r})."
                )

        # Keys are demanded only from the seams that will actually call the
        # cloud APIs — fully-local mode must not require cloud credentials.
        missing = []
        if self.llm_provider == "cloud" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if self.embedding_provider == "cloud" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.postgres_url:
            missing.append("POSTGRES_URL or DATABASE_URL")
        if missing:
            raise ValueError(
                f"Required environment variables not set: "
                f"{', '.join(missing)}. "
                f"Copy .env.example to .env and fill in all values."
            )
        return self


@lru_cache()
def get_config() -> AxiomConfig:
    """Get cached configuration instance."""
    return AxiomConfig()
