"""
Core type definitions for the model abstraction layer.

Focused on LLM calling, structured output extraction, and cost tracking.
Uses litellm.model_cost and litellm.get_model_info() for pricing/capabilities.
Data governance and additional metadata belong in the application layer.
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, SecretStr

import litellm


logger = logging.getLogger(__name__)


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get model info from litellm's database.

    Returns pricing, capabilities, context window, etc.
    Falls back to empty dict if model not found.
    """
    try:
        return litellm.get_model_info(model_id)
    except Exception:
        # Model not in litellm's database
        return {}


def get_model_cost_per_token(model_id: str) -> tuple[float, float]:
    """
    Get input/output cost per token from litellm.

    Returns (input_cost_per_token, output_cost_per_token).
    Returns (0, 0) if not found.
    """
    info = get_model_info(model_id)
    return (
        info.get("input_cost_per_token", 0) or 0,
        info.get("output_cost_per_token", 0) or 0,
    )


class ModelCredential(BaseModel):
    """Credentials for accessing a model provider."""

    provider_id: str  # "openai", "azure", "anthropic"
    api_key: SecretStr

    # Provider-specific configuration
    base_url: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None
    organization_id: Optional[str] = None  # OpenAI org

    model_config = {"extra": "allow"}


class LLMSpec(BaseModel):
    """
    Specification for an LLM model.

    Minimal spec - pricing and capabilities come from litellm.model_cost.
    """

    model_id: str  # "gpt-4o", "claude-3-5-sonnet-20241022"
    provider_id: str  # "openai", "azure", "anthropic"

    # Optional display name (defaults to model_id)
    name: Optional[str] = None

    def get_litellm_model_name(self, credential: Optional[ModelCredential] = None) -> str:
        """Get the model name for litellm routing."""
        if self.provider_id == "azure" and credential and credential.azure_deployment:
            return f"azure/{credential.azure_deployment}"
        elif self.provider_id == "anthropic":
            return f"anthropic/{self.model_id}"
        elif self.provider_id == "local":
            return f"local/{self.model_id}"
        else:
            return self.model_id

    def get_info(self) -> Dict[str, Any]:
        """Get model info from litellm."""
        return get_model_info(self.model_id)

    @property
    def context_window(self) -> int:
        """Get context window from litellm or default."""
        return self.get_info().get("max_input_tokens", 128000)

    @property
    def max_output_tokens(self) -> int:
        """Get max output tokens from litellm or default."""
        return self.get_info().get("max_output_tokens", 4096)


class EmbeddingModelSpec(BaseModel):
    """
    Specification for an embedding model.

    Minimal spec - capabilities come from litellm.model_cost.
    """

    model_id: str  # "text-embedding-3-large"
    provider_id: str  # "openai", "azure", "local"

    # Optional display name
    name: Optional[str] = None

    def get_litellm_model_name(self, credential: Optional[ModelCredential] = None) -> str:
        """Get the model name for litellm routing."""
        if self.provider_id == "local":
            return f"local/{self.model_id}"
        elif self.provider_id == "azure" and credential and credential.azure_deployment:
            return f"azure/{credential.azure_deployment}"
        else:
            return self.model_id

    def get_info(self) -> Dict[str, Any]:
        """Get model info from litellm."""
        return get_model_info(self.model_id)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions from litellm or default."""
        return self.get_info().get("output_vector_size", 1536)

    @property
    def max_input_tokens(self) -> int:
        """Get max input tokens from litellm or default."""
        return self.get_info().get("max_input_tokens", 8191)


class ModelSet(BaseModel):
    """A collection of models available for use, with credentials."""

    id: str
    name: str
    description: Optional[str] = None

    # Available models
    llms: List[LLMSpec] = Field(default_factory=list)
    embedding_models: List[EmbeddingModelSpec] = Field(default_factory=list)

    # Defaults
    default_llm_id: Optional[str] = None
    default_embedding_id: Optional[str] = None

    # Credentials (provider_id -> credential)
    credentials: Dict[str, ModelCredential] = Field(default_factory=dict)

    def get_llm(self, model_id: Optional[str] = None) -> LLMSpec:
        """Get an LLM spec by ID or return default."""
        target_id = model_id or self.default_llm_id
        if not target_id:
            if self.llms:
                return self.llms[0]
            raise ValueError("No LLMs available in ModelSet")

        for llm in self.llms:
            if llm.model_id == target_id:
                return llm
        raise ValueError(f"LLM not found: {target_id}")

    def get_embedding_model(self, model_id: Optional[str] = None) -> EmbeddingModelSpec:
        """Get an embedding model spec by ID or return default."""
        target_id = model_id or self.default_embedding_id
        if not target_id:
            if self.embedding_models:
                return self.embedding_models[0]
            raise ValueError("No embedding models available in ModelSet")

        for emb in self.embedding_models:
            if emb.model_id == target_id:
                return emb
        raise ValueError(f"Embedding model not found: {target_id}")

    def get_credential(self, provider_id: str) -> ModelCredential:
        """Get credential for a provider."""
        if provider_id not in self.credentials:
            raise ValueError(f"No credential for provider: {provider_id}")
        return self.credentials[provider_id]

    def has_credential(self, provider_id: str) -> bool:
        """Check if credential exists for a provider."""
        return provider_id in self.credentials
