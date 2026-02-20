"""
Model clients for making LLM and embedding calls.

Uses litellm Python library directly (no proxy required).
LLMClient integrates with instructor for structured outputs.
"""

import logging
from typing import List, Dict, Optional, Any, Type, TypeVar
from decimal import Decimal

import litellm
import instructor
from instructor.core.hooks import HookName
from pydantic import BaseModel

from .types import LLMSpec, EmbeddingModelSpec, ModelCredential, ModelSet


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Configure litellm globally
litellm.drop_params = True
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)


class LLMClient:
    """
    Client for making LLM calls with structured output support.

    Uses instructor + litellm internally. Credentials are passed per-call,
    not set globally, allowing multi-tenant usage.
    """

    def __init__(self, spec: LLMSpec, credential: ModelCredential):
        self.spec = spec
        self.credential = credential
        self._instructor_client = None

    def _get_litellm_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for litellm call (per-call credentials)."""
        kwargs = {
            "model": self.spec.get_litellm_model_name(self.credential),
            "api_key": self.credential.api_key.get_secret_value(),
        }

        # Base URL (for OpenAI-compatible endpoints or Azure)
        if self.credential.base_url:
            kwargs["api_base"] = self.credential.base_url

        # Azure-specific
        if self.spec.provider_id == "azure":
            kwargs["api_version"] = (
                self.credential.azure_api_version or "2024-02-15-preview"
            )

        # OpenAI organization
        if self.credential.organization_id:
            kwargs["organization"] = self.credential.organization_id

        return kwargs

    def _get_instructor_client(self):
        """Get or create instructor client."""
        if self._instructor_client is None:
            # Create instructor client wrapping litellm
            # The key insight: we pass api_key/api_base per-call via kwargs
            self._instructor_client = instructor.from_litellm(
                litellm.completion, mode=instructor.Mode.JSON
            )

            # Truncate validation errors to save tokens on retries
            def truncate_validation_errors(**kwargs):
                max_chars = 2000
                for msg in kwargs.get("messages", []):
                    content = msg.get("content", "")
                    if "validation error" in content.lower() and len(content) > max_chars:
                        msg["content"] = content[:max_chars] + "\n\n... (truncated)"

            self._instructor_client.on(HookName.COMPLETION_KWARGS, truncate_validation_errors)

        return self._instructor_client

    def completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Optional Pydantic model for structured output
            max_retries: Number of retries for validation errors
            max_tokens: Maximum tokens in response
            **kwargs: Additional kwargs passed to litellm

        Returns:
            Dict with 'response', 'completion', 'cost', 'model', 'provider'
        """
        call_kwargs = self._get_litellm_kwargs()
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)

        if response_model:
            # Use instructor for structured output
            client = self._get_instructor_client()
            response, completion = client.chat.completions.create_with_completion(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                **call_kwargs,
            )
        else:
            # Plain completion without structured output
            completion = litellm.completion(messages=messages, **call_kwargs)
            response = completion.choices[0].message.content

        # Calculate cost
        cost = self._calculate_cost(completion)

        return {
            "response": response,
            "completion": completion,
            "cost": cost,
            "model": self.spec.model_id,
            "provider": self.spec.provider_id,
        }

    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Optional Pydantic model for structured output
            max_retries: Number of retries for validation errors
            max_tokens: Maximum tokens in response
            **kwargs: Additional kwargs passed to litellm

        Returns:
            Dict with 'response', 'completion', 'cost', 'model', 'provider'
        """
        call_kwargs = self._get_litellm_kwargs()
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)

        if response_model:
            # Use instructor for structured output
            # Note: instructor's from_litellm with async needs acompletion
            client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
            response, completion = await client.chat.completions.create_with_completion(
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                **call_kwargs,
            )
        else:
            # Plain async completion
            completion = await litellm.acompletion(messages=messages, **call_kwargs)
            response = completion.choices[0].message.content

        # Calculate cost
        cost = self._calculate_cost(completion)

        return {
            "response": response,
            "completion": completion,
            "cost": cost,
            "model": self.spec.model_id,
            "provider": self.spec.provider_id,
        }

    def _calculate_cost(self, completion) -> float:
        """Calculate cost from completion response."""
        try:
            return litellm.completion_cost(completion_response=completion)
        except Exception:
            # Fallback to spec-based estimation
            return self._estimate_cost(completion)

    def _estimate_cost(self, completion) -> float:
        """Fallback cost estimation using spec pricing."""
        try:
            usage = completion.usage
            input_cost = (usage.prompt_tokens / 1000) * float(self.spec.input_cost_per_1k)
            output_cost = (usage.completion_tokens / 1000) * float(self.spec.output_cost_per_1k)
            return input_cost + output_cost
        except Exception:
            return 0.0


class EmbeddingClient:
    """
    Client for making embedding calls.

    Supports both API-based embeddings (OpenAI, Azure) and local
    sentence-transformers models.
    """

    def __init__(self, spec: EmbeddingModelSpec, credential: Optional[ModelCredential]):
        self.spec = spec
        self.credential = credential

    def _is_local(self) -> bool:
        """Check if this is a local model."""
        return self.spec.provider_id == "local"

    def _get_litellm_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for litellm embedding call."""
        kwargs = {
            "model": self.spec.get_litellm_model_name(self.credential),
        }

        if not self._is_local() and self.credential:
            kwargs["api_key"] = self.credential.api_key.get_secret_value()
            if self.credential.base_url:
                kwargs["api_base"] = self.credential.base_url
            if self.spec.provider_id == "azure" and self.credential.azure_api_version:
                kwargs["api_version"] = self.credential.azure_api_version

        return kwargs

    def embed(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get embeddings for texts (sync).

        Args:
            texts: List of texts to embed
            dimensions: Optional dimensions (for models that support it)
            **kwargs: Additional kwargs passed to litellm

        Returns:
            Dict with 'embeddings', 'cost', 'model', 'provider'
        """
        call_kwargs = self._get_litellm_kwargs()
        if dimensions:
            call_kwargs["dimensions"] = dimensions
        call_kwargs.update(kwargs)

        response = litellm.embedding(input=texts, **call_kwargs)

        # Extract embeddings
        embeddings = [d["embedding"] for d in response.data]

        # Calculate cost
        cost = self._calculate_cost(response, len(texts))

        return {
            "embeddings": embeddings,
            "cost": cost,
            "model": self.spec.model_id,
            "provider": self.spec.provider_id,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    async def aembed(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get embeddings for texts (async).

        Args:
            texts: List of texts to embed
            dimensions: Optional dimensions (for models that support it)
            **kwargs: Additional kwargs passed to litellm

        Returns:
            Dict with 'embeddings', 'cost', 'model', 'provider'
        """
        call_kwargs = self._get_litellm_kwargs()
        if dimensions:
            call_kwargs["dimensions"] = dimensions
        call_kwargs.update(kwargs)

        response = await litellm.aembedding(input=texts, **call_kwargs)

        # Extract embeddings
        embeddings = [d["embedding"] for d in response.data]

        # Calculate cost
        cost = self._calculate_cost(response, len(texts))

        return {
            "embeddings": embeddings,
            "cost": cost,
            "model": self.spec.model_id,
            "provider": self.spec.provider_id,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    def _calculate_cost(self, response, num_texts: int) -> Optional[float]:
        """Calculate cost from embedding response."""
        try:
            return litellm.completion_cost(
                completion_response=response,
                call_type="embedding",
            )
        except Exception:
            return self._estimate_cost(num_texts)

    def _estimate_cost(self, num_texts: int) -> Optional[float]:
        """Fallback cost estimation using spec pricing."""
        if self.spec.cost_per_1k_tokens is None:
            return None
        # Rough estimate: ~250 tokens per text chunk
        estimated_tokens = num_texts * 250
        return (estimated_tokens / 1000) * float(self.spec.cost_per_1k_tokens)


class ModelClient:
    """
    Unified client for a ModelSet.

    Provides easy access to LLM and embedding clients with automatic
    credential resolution.
    """

    def __init__(self, model_set: ModelSet):
        self.model_set = model_set
        self._llm_clients: Dict[str, LLMClient] = {}
        self._embedding_clients: Dict[str, EmbeddingClient] = {}

    def llm(self, model_id: Optional[str] = None) -> LLMClient:
        """
        Get LLM client for a model (cached).

        Args:
            model_id: Model ID or None for default

        Returns:
            LLMClient for the model
        """
        spec = self.model_set.get_llm(model_id)
        if spec.model_id not in self._llm_clients:
            credential = self.model_set.get_credential(spec.provider_id)
            self._llm_clients[spec.model_id] = LLMClient(spec, credential)
        return self._llm_clients[spec.model_id]

    def embedding(self, model_id: Optional[str] = None) -> EmbeddingClient:
        """
        Get embedding client for a model (cached).

        Args:
            model_id: Model ID or None for default

        Returns:
            EmbeddingClient for the model
        """
        spec = self.model_set.get_embedding_model(model_id)
        if spec.model_id not in self._embedding_clients:
            credential = (
                self.model_set.credentials.get(spec.provider_id)
                if spec.provider_id != "local"
                else None
            )
            self._embedding_clients[spec.model_id] = EmbeddingClient(spec, credential)
        return self._embedding_clients[spec.model_id]

    def completion(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for completion using default or specified model."""
        return self.llm(model_id).completion(messages, **kwargs)

    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for async completion."""
        return await self.llm(model_id).acompletion(messages, **kwargs)

    def embed(
        self,
        texts: List[str],
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for embedding."""
        return self.embedding(model_id).embed(texts, **kwargs)

    async def aembed(
        self,
        texts: List[str],
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for async embedding."""
        return await self.embedding(model_id).aembed(texts, **kwargs)
