import os
from unittest.mock import patch

from axiom.config import AxiomConfig


def test_config_embedding_dimensions_default_is_1536():
    cfg = AxiomConfig()
    assert cfg.embedding_dimensions == 1536


def test_config_embedding_dimensions_env_override():
    overrides = {
        "EMBEDDING_DIMENSIONS": "512",
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "POSTGRES_URL": "postgresql+psycopg://test:test@localhost:5432/test",
    }
    with patch.dict(os.environ, overrides):
        cfg = AxiomConfig()
        assert cfg.embedding_dimensions == 512
