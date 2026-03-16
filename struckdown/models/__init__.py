"""
Model abstraction layer for LLMs and embedding models.

Provides a clean abstraction over different model providers (OpenAI, Azure,
Anthropic, local) without requiring a LiteLLM proxy. Uses the litellm Python
library directly for API calls.

Focused on:
- LLM calling with structured output (via instructor)
- Embedding generation
- Cost tracking

Data governance and additional metadata belong in the application layer.

Key classes:
- ModelCredential: API credentials for a provider
- LLMSpec: Specification for an LLM (model ID, pricing)
- EmbeddingModelSpec: Specification for an embedding model
- ModelSet: Collection of models + credentials
- LLMClient: Client for making LLM calls (with instructor)
- EmbeddingClient: Client for making embedding calls
- ModelClient: Unified client for a ModelSet
"""

from .clients import EmbeddingClient, LLMClient, ModelClient
from .defaults import (ANTHROPIC_LLMS, EMBEDDING_MODELS, OPENAI_LLMS,
                       PROVIDER_ANTHROPIC, PROVIDER_AZURE, PROVIDER_LOCAL,
                       PROVIDER_OPENAI, create_model_set_from_env,
                       create_openai_model_set)
from .types import EmbeddingModelSpec, LLMSpec, ModelCredential, ModelSet

__all__ = [
    # Types
    "ModelCredential",
    "LLMSpec",
    "EmbeddingModelSpec",
    "ModelSet",
    # Clients
    "LLMClient",
    "EmbeddingClient",
    "ModelClient",
    # Provider IDs
    "PROVIDER_OPENAI",
    "PROVIDER_AZURE",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_LOCAL",
    # Default specs
    "OPENAI_LLMS",
    "ANTHROPIC_LLMS",
    "EMBEDDING_MODELS",
    # Factory functions
    "create_openai_model_set",
    "create_model_set_from_env",
]
