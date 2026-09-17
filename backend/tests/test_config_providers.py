"""Provider-mode config tests (W3): provider-aware required-key validation.

Cloud mode (the default) must demand exactly the keys it always did — CI and
tests rely on dummy keys satisfying the validator. Fully-local mode must NOT
demand cloud credentials, or "fully-local" would still be gated on a cloud
account. Both branches are asserted, plus provider-name validation.
"""

import pytest
from pydantic import ValidationError

from axiom.config import AxiomConfig

_PG = "postgresql+psycopg://test:test@localhost:5432/test"


def _cloud_config() -> AxiomConfig:
    return AxiomConfig(
        anthropic_api_key="sk-ant-x",
        openai_api_key="sk-x",
        postgres_url=_PG,
    )


def _local_config() -> AxiomConfig:
    return AxiomConfig(
        llm_provider="local",
        embedding_provider="local",
        ollama_host="http://localhost:11434",
        anthropic_api_key="",
        openai_api_key="",
        postgres_url=_PG,
    )


class TestCloudModeValidation:
    def test_cloud_mode_with_keys_passes(self):
        cfg = _cloud_config()
        assert cfg.llm_provider == "cloud"
        assert cfg.embedding_provider == "cloud"

    def test_cloud_mode_requires_anthropic_key(self):
        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            AxiomConfig(anthropic_api_key="", openai_api_key="sk-x", postgres_url=_PG)

    def test_cloud_mode_requires_openai_key(self):
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            AxiomConfig(anthropic_api_key="sk-ant-x", openai_api_key="", postgres_url=_PG)

    def test_cloud_mode_still_requires_postgres(self):
        with pytest.raises(ValidationError, match="POSTGRES_URL"):
            AxiomConfig(anthropic_api_key="sk-ant-x", openai_api_key="sk-x", postgres_url="")


class TestLocalModeValidation:
    def test_local_mode_needs_no_cloud_keys(self):
        """The blocker for fully-local mode: no cloud key demand, ever."""
        cfg = _local_config()
        assert cfg.anthropic_api_key == ""
        assert cfg.openai_api_key == ""

    def test_local_mode_still_requires_postgres(self):
        """Local mode drops cloud keys but keeps Postgres required (pgvector,
        traces, and the checkpointer all live there)."""
        with pytest.raises(ValidationError, match="POSTGRES_URL"):
            AxiomConfig(
                llm_provider="local",
                embedding_provider="local",
                anthropic_api_key="",
                openai_api_key="",
                postgres_url="",
            )

    def test_mixed_mode_demands_only_cloud_seam_keys(self):
        """Local generation with cloud embeddings still needs the OpenAI key
        (embeddings remain cloud), but not the Anthropic key."""
        cfg = AxiomConfig(
            llm_provider="local",
            embedding_provider="cloud",
            openai_api_key="sk-x",
            anthropic_api_key="",
            postgres_url=_PG,
        )
        assert cfg.llm_provider == "local"
        assert cfg.embedding_provider == "cloud"


class TestProviderNameValidation:
    def test_unknown_llm_provider_rejected(self):
        with pytest.raises(ValidationError, match="LLM_PROVIDER"):
            AxiomConfig(
                anthropic_api_key="k", openai_api_key="k", postgres_url=_PG, llm_provider="banana"
            )

    def test_unknown_embedding_provider_rejected(self):
        with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER"):
            AxiomConfig(
                anthropic_api_key="k", openai_api_key="k", postgres_url=_PG, embedding_provider="banana"
            )


class TestEffectiveEmbeddingGeometry:
    def test_cloud_defaults_unchanged(self):
        cfg = _cloud_config()
        assert cfg.effective_embedding_model == "text-embedding-3-small"
        assert cfg.effective_embedding_dimensions == 1536

    def test_local_mode_uses_native_model_and_width(self):
        cfg = _local_config()
        assert cfg.effective_embedding_model == "nomic-embed-text"
        assert cfg.effective_embedding_dimensions == 768

    def test_local_dimensions_env_override(self):
        cfg = AxiomConfig(
            llm_provider="local",
            embedding_provider="local",
            local_embedding_dimensions=512,
            anthropic_api_key="",
            openai_api_key="",
            postgres_url=_PG,
        )
        assert cfg.effective_embedding_dimensions == 512

    def test_generator_model_defaults_distinct_from_critic(self):
        """critic != generator by default: same-model self-grading correlates
        errors (locked decision D4)."""
        cfg = _local_config()
        assert cfg.ollama_generation_model != cfg.ollama_critic_model
