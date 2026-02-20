"""
Tests for the model abstraction layer.

Tests instructor integration, structured outputs, and embedding calls.
"""

import os
import pytest
from decimal import Decimal
from typing import List
from pydantic import BaseModel, Field

from struckdown.models import (
    ModelCredential,
    LLMSpec,
    EmbeddingModelSpec,
    ModelSet,
    LLMClient,
    EmbeddingClient,
    ModelClient,
    create_openai_model_set,
    create_model_set_from_env,
    OPENAI_LLMS,
    EMBEDDING_MODELS,
)
from pydantic import SecretStr


# Test response models for structured output
class SimpleResponse(BaseModel):
    """Simple response for testing."""
    answer: str
    confidence: float = Field(ge=0, le=1)


class ListResponse(BaseModel):
    """List response for testing."""
    items: List[str]
    count: int


class AnalysisResponse(BaseModel):
    """More complex response for testing."""
    summary: str
    key_points: List[str]
    sentiment: str = Field(description="positive, negative, or neutral")


# Fixtures
@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        pytest.skip("No OpenAI API key available (set OPENAI_API_KEY or LLM_API_KEY)")
    return key


@pytest.fixture
def openai_credential(openai_api_key):
    """Create OpenAI credential."""
    return ModelCredential(
        provider_id="openai",
        api_key=SecretStr(openai_api_key),
        base_url=os.getenv("LLM_API_BASE"),  # Optional, for proxy
    )


@pytest.fixture
def gpt4o_mini_spec():
    """GPT-4.1-mini spec (available on proxy)."""
    return LLMSpec(
        model_id="gpt-4.1-mini",
        provider_id="openai",
        name="GPT-4.1 Mini",
        context_window=1000000,
        max_output_tokens=32768,
        input_cost_per_1k=Decimal("0.0004"),
        output_cost_per_1k=Decimal("0.0016"),
    )


@pytest.fixture
def embedding_spec():
    """Embedding model spec."""
    return EmbeddingModelSpec(
        model_id="text-embedding-3-small",
        provider_id="openai",
        name="OpenAI Embedding Small",
        dimensions=1536,
        max_input_tokens=8191,
        cost_per_1k_tokens=Decimal("0.00002"),
    )


@pytest.fixture
def llm_client(gpt4o_mini_spec, openai_credential):
    """Create LLM client."""
    return LLMClient(gpt4o_mini_spec, openai_credential)


@pytest.fixture
def embedding_client(embedding_spec, openai_credential):
    """Create embedding client."""
    return EmbeddingClient(embedding_spec, openai_credential)


@pytest.fixture
def model_set(openai_api_key, gpt4o_mini_spec, embedding_spec):
    """Create a model set with proxy-compatible models."""
    from pydantic import SecretStr

    credential = ModelCredential(
        provider_id="openai",
        api_key=SecretStr(openai_api_key),
        base_url=os.getenv("LLM_API_BASE"),
    )

    return ModelSet(
        id="test-set",
        name="Test Model Set",
        llms=[gpt4o_mini_spec],
        embedding_models=[embedding_spec],
        default_llm_id=gpt4o_mini_spec.model_id,
        default_embedding_id=embedding_spec.model_id,
        credentials={"openai": credential},
    )


@pytest.fixture
def model_client(model_set):
    """Create model client."""
    return ModelClient(model_set)


# Unit tests (no API calls)
class TestModelTypes:
    """Test model type definitions."""

    def test_model_credential_creation(self):
        """Test creating model credentials."""
        cred = ModelCredential(
            provider_id="openai",
            api_key=SecretStr("test-key"),
            base_url="https://api.example.com",
        )
        assert cred.provider_id == "openai"
        assert cred.api_key.get_secret_value() == "test-key"
        assert cred.base_url == "https://api.example.com"

    def test_llm_spec_litellm_name_openai(self):
        """Test LLM spec generates correct litellm model name for OpenAI."""
        spec = LLMSpec(
            model_id="gpt-4o",
            provider_id="openai",
            name="GPT-4o",
        )
        cred = ModelCredential(provider_id="openai", api_key=SecretStr("key"))
        assert spec.get_litellm_model_name(cred) == "gpt-4o"

    def test_llm_spec_litellm_name_anthropic(self):
        """Test LLM spec generates correct litellm model name for Anthropic."""
        spec = LLMSpec(
            model_id="claude-3-5-sonnet-20241022",
            provider_id="anthropic",
            name="Claude 3.5 Sonnet",
        )
        cred = ModelCredential(provider_id="anthropic", api_key=SecretStr("key"))
        assert spec.get_litellm_model_name(cred) == "anthropic/claude-3-5-sonnet-20241022"

    def test_llm_spec_litellm_name_azure(self):
        """Test LLM spec generates correct litellm model name for Azure."""
        spec = LLMSpec(
            model_id="gpt-4o",
            provider_id="azure",
            name="Azure GPT-4o",
        )
        cred = ModelCredential(
            provider_id="azure",
            api_key=SecretStr("key"),
            azure_deployment="my-deployment",
        )
        assert spec.get_litellm_model_name(cred) == "azure/my-deployment"

    def test_model_set_get_llm(self):
        """Test getting LLM from model set."""
        llm1 = LLMSpec(model_id="model1", provider_id="openai", name="Model 1")
        llm2 = LLMSpec(model_id="model2", provider_id="openai", name="Model 2")

        model_set = ModelSet(
            id="test",
            name="Test Set",
            llms=[llm1, llm2],
            default_llm_id="model1",
            credentials={
                "openai": ModelCredential(provider_id="openai", api_key=SecretStr("key"))
            },
        )

        # Get default
        assert model_set.get_llm().model_id == "model1"
        # Get specific
        assert model_set.get_llm("model2").model_id == "model2"

    def test_model_set_get_credential(self):
        """Test getting credential from model set."""
        model_set = ModelSet(
            id="test",
            name="Test Set",
            credentials={
                "openai": ModelCredential(provider_id="openai", api_key=SecretStr("key1")),
                "anthropic": ModelCredential(provider_id="anthropic", api_key=SecretStr("key2")),
            },
        )

        assert model_set.get_credential("openai").api_key.get_secret_value() == "key1"
        assert model_set.get_credential("anthropic").api_key.get_secret_value() == "key2"

    def test_create_openai_model_set(self):
        """Test creating OpenAI model set."""
        model_set = create_openai_model_set(api_key="test-key")

        assert model_set.id == "openai-default"
        assert len(model_set.llms) > 0
        assert len(model_set.embedding_models) > 0
        assert model_set.has_credential("openai")


# Integration tests (require API key)
@pytest.mark.integration
class TestLLMClientIntegration:
    """Integration tests for LLM client with real API calls."""

    def test_simple_completion(self, llm_client):
        """Test simple completion without structured output."""
        result = llm_client.completion(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )

        assert "response" in result
        assert "completion" in result
        assert "cost" in result
        assert result["model"] == "gpt-4.1-mini"
        assert result["provider"] == "openai"
        assert "hello" in result["response"].lower()

    def test_structured_output_simple(self, llm_client):
        """Test structured output with simple model."""
        result = llm_client.completion(
            messages=[
                {"role": "user", "content": "What is 2+2? Be confident."}
            ],
            response_model=SimpleResponse,
            max_tokens=100,
        )

        assert isinstance(result["response"], SimpleResponse)
        assert "4" in result["response"].answer or "four" in result["response"].answer.lower()
        assert 0 <= result["response"].confidence <= 1
        assert result["cost"] > 0

    def test_structured_output_list(self, llm_client):
        """Test structured output with list response."""
        result = llm_client.completion(
            messages=[
                {"role": "user", "content": "List exactly 3 primary colors."}
            ],
            response_model=ListResponse,
            max_tokens=100,
        )

        assert isinstance(result["response"], ListResponse)
        assert len(result["response"].items) == 3
        assert result["response"].count == 3

    def test_structured_output_complex(self, llm_client):
        """Test structured output with complex model."""
        result = llm_client.completion(
            messages=[
                {
                    "role": "user",
                    "content": "Analyze this text: 'I love sunny days! They make me so happy.'",
                }
            ],
            response_model=AnalysisResponse,
            max_tokens=200,
        )

        assert isinstance(result["response"], AnalysisResponse)
        assert len(result["response"].summary) > 0
        assert len(result["response"].key_points) > 0
        assert result["response"].sentiment in ["positive", "negative", "neutral"]


@pytest.mark.integration
class TestEmbeddingClientIntegration:
    """Integration tests for embedding client."""

    def test_single_embedding(self, embedding_client, embedding_spec):
        """Test getting a single embedding."""
        result = embedding_client.embed(texts=["Hello, world!"])

        assert "embeddings" in result
        assert len(result["embeddings"]) == 1
        # Dimension should be > 0 (actual dimension depends on model)
        assert len(result["embeddings"][0]) > 0
        assert result["model"] == embedding_spec.model_id

    def test_batch_embeddings(self, embedding_client):
        """Test getting multiple embeddings."""
        texts = ["First text", "Second text", "Third text"]
        result = embedding_client.embed(texts=texts)

        assert len(result["embeddings"]) == 3
        for emb in result["embeddings"]:
            assert len(emb) > 0  # Just check embeddings exist

    def test_embedding_cost_tracking(self, embedding_client):
        """Test that embedding calls track cost."""
        result = embedding_client.embed(texts=["Test text for cost tracking"])

        # Cost should be present (may be None for some providers)
        assert "cost" in result


@pytest.mark.integration
class TestModelClientIntegration:
    """Integration tests for unified model client."""

    def test_model_client_completion(self, model_client):
        """Test completion via model client."""
        result = model_client.completion(
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=10,
        )

        assert "response" in result
        assert "test" in result["response"].lower()

    def test_model_client_structured(self, model_client):
        """Test structured output via model client."""
        result = model_client.completion(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_model=SimpleResponse,
            max_tokens=100,
        )

        assert isinstance(result["response"], SimpleResponse)
        assert "paris" in result["response"].answer.lower()

    def test_model_client_embedding(self, model_client):
        """Test embedding via model client."""
        result = model_client.embed(texts=["Test embedding"])

        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) > 0


@pytest.mark.integration
class TestAsyncIntegration:
    """Async integration tests using anyio."""

    def test_async_completion(self, llm_client):
        """Test async completion."""
        import anyio

        async def run_test():
            result = await llm_client.acompletion(
                messages=[{"role": "user", "content": "Say 'async' and nothing else."}],
                max_tokens=10,
            )
            assert "async" in result["response"].lower()

        anyio.run(run_test)

    def test_async_structured_output(self, llm_client):
        """Test async structured output."""
        import anyio

        async def run_test():
            result = await llm_client.acompletion(
                messages=[{"role": "user", "content": "What is 3+3?"}],
                response_model=SimpleResponse,
                max_tokens=100,
            )
            assert isinstance(result["response"], SimpleResponse)
            assert "6" in result["response"].answer

        anyio.run(run_test)

    def test_async_embedding(self, embedding_client):
        """Test async embedding."""
        import anyio

        async def run_test():
            result = await embedding_client.aembed(texts=["Async embedding test"])
            assert len(result["embeddings"]) == 1

        anyio.run(run_test)


# Test that model set from env works
class TestModelSetFromEnv:
    """Test creating model set from environment."""

    def test_create_from_env_with_openai(self, monkeypatch):
        """Test creating model set when OPENAI_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model_set = create_model_set_from_env()

        assert model_set.has_credential("openai")
        assert len(model_set.llms) > 0

    def test_create_from_env_with_llm_vars(self, monkeypatch):
        """Test creating model set with LLM_API_KEY fallback."""
        # Clear any existing keys
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")

        model_set = create_model_set_from_env()

        assert model_set.has_credential("openai")
        cred = model_set.get_credential("openai")
        assert cred.base_url == "https://api.example.com/v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
