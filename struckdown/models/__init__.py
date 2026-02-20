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

from .types import (
    ModelCredential,
    LLMSpec,
    EmbeddingModelSpec,
    ModelSet,
)
from .clients import (
    LLMClient,
    EmbeddingClient,
    ModelClient,
)
from .defaults import (
    PROVIDER_OPENAI,
    PROVIDER_AZURE,
    PROVIDER_ANTHROPIC,
    PROVIDER_LOCAL,
    OPENAI_LLMS,
    ANTHROPIC_LLMS,
    EMBEDDING_MODELS,
    create_openai_model_set,
    create_model_set_from_env,
)

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
