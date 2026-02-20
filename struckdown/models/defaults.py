"""
Default model specs and factory functions for common providers.

Pricing and capabilities come from litellm.model_cost - we just define
which models to include by default.
"""

import os
from typing import Optional, Dict, List

from pydantic import SecretStr

from .types import (
    ModelCredential,
    LLMSpec,
    EmbeddingModelSpec,
    ModelSet,
)


# Provider IDs
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE = "azure"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_LOCAL = "local"


def llm(model_id: str, provider_id: str = "openai", name: Optional[str] = None) -> LLMSpec:
    """Helper to create an LLMSpec."""
    return LLMSpec(model_id=model_id, provider_id=provider_id, name=name)


def emb(model_id: str, provider_id: str = "openai", name: Optional[str] = None) -> EmbeddingModelSpec:
    """Helper to create an EmbeddingModelSpec."""
    return EmbeddingModelSpec(model_id=model_id, provider_id=provider_id, name=name)


# Common OpenAI LLMs (pricing/capabilities from litellm.model_cost)
OPENAI_LLMS = [
    llm("gpt-4o", "openai", "GPT-4o"),
    llm("gpt-4o-mini", "openai", "GPT-4o Mini"),
    llm("gpt-4.1-mini", "openai", "GPT-4.1 Mini"),
    llm("o1", "openai", "o1"),
    llm("o1-mini", "openai", "o1 Mini"),
]

# Common Anthropic LLMs
ANTHROPIC_LLMS = [
    llm("claude-sonnet-4-20250514", "anthropic", "Claude Sonnet 4"),
    llm("claude-3-5-sonnet-20241022", "anthropic", "Claude 3.5 Sonnet"),
    llm("claude-3-5-haiku-20241022", "anthropic", "Claude 3.5 Haiku"),
]

# Common embedding models
EMBEDDING_MODELS = [
    emb("text-embedding-3-large", "openai", "OpenAI Large"),
    emb("text-embedding-3-small", "openai", "OpenAI Small"),
    emb("intfloat/e5-large-v2", "local", "E5 Large (Local)"),
    emb("all-MiniLM-L6-v2", "local", "MiniLM (Local)"),
]


def create_openai_model_set(
    api_key: str,
    base_url: Optional[str] = None,
    organization_id: Optional[str] = None,
    llms: Optional[List[LLMSpec]] = None,
    embedding_models: Optional[List[EmbeddingModelSpec]] = None,
) -> ModelSet:
    """Create a ModelSet with OpenAI models."""
    credential = ModelCredential(
        provider_id="openai",
        api_key=SecretStr(api_key),
        base_url=base_url,
        organization_id=organization_id,
    )

    return ModelSet(
        id="openai-default",
        name="OpenAI Models",
        llms=llms or OPENAI_LLMS,
        embedding_models=embedding_models or [em for em in EMBEDDING_MODELS if em.provider_id == "openai"],
        default_llm_id="gpt-4o-mini",
        default_embedding_id="text-embedding-3-small",
        credentials={"openai": credential},
    )


def create_anthropic_model_set(
    api_key: str,
    llms: Optional[List[LLMSpec]] = None,
) -> ModelSet:
    """Create a ModelSet with Anthropic models."""
    credential = ModelCredential(
        provider_id="anthropic",
        api_key=SecretStr(api_key),
    )

    return ModelSet(
        id="anthropic-default",
        name="Anthropic Models",
        llms=llms or ANTHROPIC_LLMS,
        embedding_models=[],
        default_llm_id="claude-sonnet-4-20250514",
        credentials={"anthropic": credential},
    )


def create_model_set_from_env(
    openai_key_var: str = "OPENAI_API_KEY",
    anthropic_key_var: str = "ANTHROPIC_API_KEY",
    llm_api_key_var: str = "LLM_API_KEY",
    llm_api_base_var: str = "LLM_API_BASE",
) -> ModelSet:
    """
    Create a ModelSet from environment variables.

    Checks for provider-specific keys first, then falls back to
    LLM_API_KEY/LLM_API_BASE for OpenAI-compatible usage.
    """
    credentials: Dict[str, ModelCredential] = {}
    llms: list = []
    embeddings: list = []

    # Check for OpenAI
    openai_key = os.getenv(openai_key_var)
    if openai_key:
        credentials["openai"] = ModelCredential(
            provider_id="openai",
            api_key=SecretStr(openai_key),
        )
        llms.extend(OPENAI_LLMS)
        embeddings.extend([em for em in EMBEDDING_MODELS if em.provider_id == "openai"])

    # Check for Anthropic
    anthropic_key = os.getenv(anthropic_key_var)
    if anthropic_key:
        credentials["anthropic"] = ModelCredential(
            provider_id="anthropic",
            api_key=SecretStr(anthropic_key),
        )
        llms.extend(ANTHROPIC_LLMS)

    # Fallback: LLM_API_KEY/LLM_API_BASE (OpenAI-compatible)
    llm_key = os.getenv(llm_api_key_var)
    llm_base = os.getenv(llm_api_base_var)
    if llm_key and "openai" not in credentials:
        credentials["openai"] = ModelCredential(
            provider_id="openai",
            api_key=SecretStr(llm_key),
            base_url=llm_base,
        )
        llms.extend(OPENAI_LLMS)
        embeddings.extend([em for em in EMBEDDING_MODELS if em.provider_id == "openai"])

    # Always include local embedding option
    embeddings.extend([em for em in EMBEDDING_MODELS if em.provider_id == "local"])

    # Determine defaults
    default_llm = llms[0].model_id if llms else None
    default_embedding = "text-embedding-3-small" if "openai" in credentials else None
    if not default_embedding and embeddings:
        default_embedding = embeddings[0].model_id

    return ModelSet(
        id="env-default",
        name="Environment Models",
        llms=llms,
        embedding_models=embeddings,
        default_llm_id=default_llm,
        default_embedding_id=default_embedding,
        credentials=credentials,
    )
