"""AXIOM Configuration - All environment variables and settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


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

    # API limits
    max_query_length: int = 2000
    max_ingest_size_mb: int = 50
    rate_limit_per_minute: int = 30
    
    # Ollama (local critic)
    ollama_host: str = "http://localhost:11434"
    ollama_critic_model: str = "llama3.2"

    # Evaluation backend
    use_claude_evaluator: bool = True   # False to use local Ollama instead
    
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

    @model_validator(mode='after')
    def validate_required_keys(self) -> 'AxiomConfig':
        missing = []
        if not self.anthropic_api_key:
            missing.append('ANTHROPIC_API_KEY')
        if not self.openai_api_key:
            missing.append('OPENAI_API_KEY')
        if not self.postgres_url:
            missing.append('POSTGRES_URL or DATABASE_URL')
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
