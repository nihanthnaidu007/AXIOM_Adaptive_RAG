"""Pure provider-mode helpers shared by config and Alembic migrations.

Alembic migrations must not import ``axiom.config``: instantiating
``AxiomConfig`` runs the required-keys validator, and the fresh-database CI
job intentionally sets no model keys. This module carries only constants and
pure functions so the runtime config and the W3 embedding-dimension migration
resolve provider geometry identically.
"""

CLOUD_PROVIDER = "cloud"
LOCAL_PROVIDER = "local"
PROVIDERS = (CLOUD_PROVIDER, LOCAL_PROVIDER)

# Defaults mirrored by AxiomConfig fields. The W3 migration reads the same
# env vars with the same defaults; keep the two in sync (see
# alembic/versions for the cross-reference).
DEFAULT_CLOUD_DIMENSIONS = 1536
DEFAULT_LOCAL_DIMENSIONS = 768


def normalize_provider(provider: str) -> str:
    """Validate a provider name; returns the normalized value or raises."""
    normalized = (provider or "").strip().lower()
    if normalized not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r} — expected one of {', '.join(PROVIDERS)}")
    return normalized


def resolve_embedding_dimensions(
    provider: str, cloud_dimensions: int, local_dimensions: int
) -> int:
    """Effective pgvector width for a provider mode.

    Local mode owns its dimension explicitly: Ollama's OpenAI-compat layer
    ignores OpenAI's ``dimensions`` kwarg, so the model's native width is
    configured (and enforced in code) instead of requested per call.
    """
    return local_dimensions if normalize_provider(provider) == LOCAL_PROVIDER else cloud_dimensions
