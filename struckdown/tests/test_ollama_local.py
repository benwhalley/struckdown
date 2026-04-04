"""Tests for local Ollama integration with pydantic-ai backend.

Requires a running Ollama instance with a qwen3 model (e.g. qwen3:4b or qwen3:8b).
Skips with a warning if Ollama is not available or the model is missing.

Note: small local models can be unreliable with bool slots -- qwen3 sometimes returns
verbose text instead of using the tool call for boolean extraction. The tests here are
chosen to work reliably with qwen3:8b but may be flaky with smaller variants.
"""

import warnings
from typing import Optional

import pytest

from struckdown import chatter
from struckdown.llm import LLM, LLMCredentials

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"


def _find_qwen3_model() -> Optional[str]:
    """Check Ollama is running and find a qwen3 model. Returns model name or None."""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        for m in models:
            name = m.get("name", "")
            if "qwen3" in name.lower():
                return name
    except Exception:
        pass
    return None


_qwen3_model = _find_qwen3_model()

if _qwen3_model is None:
    warnings.warn(
        "\n\n*** OLLAMA NOT AVAILABLE or no qwen3 model found ***\n"
        "Install Ollama and run: ollama pull qwen3:8b\n"
        "All Ollama tests will be skipped.\n",
        stacklevel=1,
    )

requires_ollama = pytest.mark.skipif(
    _qwen3_model is None,
    reason="Ollama not running or no qwen3 model available",
)


@pytest.fixture
def llm():
    return LLM(model_name=_qwen3_model)


@pytest.fixture
def creds():
    return LLMCredentials(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)


@requires_ollama
@pytest.mark.ollama
class TestOllamaStruckdown:
    """Test struckdown templates via Ollama with qwen3."""

    def test_simple_text_completion(self, llm, creds):
        """Basic text slot."""
        result = chatter(
            "What is the capital of France? [[answer]]",
            model=llm,
            credentials=creds,
        )
        assert "paris" in str(result.response).lower()

    def test_bool_slot(self, llm, creds):
        """Boolean slot -- exercises tool-to-prompted fallback on models that
        return text instead of a tool call for simple yes/no questions."""
        result = chatter(
            "Is the Earth round? [[bool:is_round]]",
            model=llm,
            credentials=creds,
        )
        assert result.outputs.is_round is True

    def test_int_slot(self, llm, creds):
        """Integer slot extraction."""
        result = chatter(
            "How many legs does a dog have? [[int:legs]]",
            model=llm,
            credentials=creds,
        )
        assert result.outputs.legs == 4

    def test_pick_slot(self, llm, creds):
        """Pick from options (enum-like)."""
        result = chatter(
            "Classify the sentiment: 'I absolutely love this product!' "
            "[[pick:sentiment|positive,negative,neutral]]",
            model=llm,
            credentials=creds,
        )
        assert result.outputs.sentiment == "positive"

    def test_pick_list_extraction(self, llm, creds):
        """Extract a list of items using quantifier syntax."""
        result = chatter(
            "Name 3 primary colours used in painting: "
            "[[pick{3}:colours|red,blue,yellow,green,orange,purple]]",
            model=llm,
            credentials=creds,
        )
        colours = result.outputs.colours
        assert len(colours) == 3
        for c in colours:
            assert c in ("red", "blue", "yellow", "green", "orange", "purple")

    def test_multi_slot_template(self, llm, creds):
        """Template with multiple dependent slots across checkpoints."""
        template = (
            "What is the largest planet in our solar system? [[planet]]\n\n"
            "<checkpoint>\n\n"
            "In one word, what colour is {{planet}} when viewed from space? [[colour]]"
        )
        result = chatter(template, model=llm, credentials=creds)
        assert "jupiter" in str(result.outputs.planet).lower()
        assert len(str(result.outputs.colour)) > 0

    def test_context_variable_with_pick(self, llm, creds):
        """Template with Jinja2 context variables and pick slot."""
        result = chatter(
            "{{text}}\n\nWhat is the overall sentiment of this text? "
            "[[pick:sentiment|positive,negative,neutral,mixed]]",
            context={"text": "The weather was absolutely terrible and I hated every minute."},
            model=llm,
            credentials=creds,
        )
        assert result.outputs.sentiment == "negative"

    def test_system_message(self, llm, creds):
        """Template with <system> tag."""
        result = chatter(
            "<system>You are a helpful assistant. Always respond in one word.</system>\n\n"
            "What colour is grass? [[answer]]",
            model=llm,
            credentials=creds,
        )
        assert "green" in str(result.response).lower()

    def test_number_slot(self, llm, creds):
        """Numeric slot extraction."""
        result = chatter(
            "How many continents are there on Earth? [[int:count]]",
            model=llm,
            credentials=creds,
        )
        assert result.outputs.count == 7

    def test_two_picks_in_one_segment(self, llm, creds):
        """Two pick slots in a single segment (parallel extraction)."""
        result = chatter(
            "'I bought a red dress.'\n\n"
            "What colour is mentioned? [[pick:colour|red,blue,green,yellow]]\n"
            "What item is mentioned? [[pick:item|hat,dress,shoes,bag]]",
            model=llm,
            credentials=creds,
        )
        assert result.outputs.colour == "red"
        assert result.outputs.item == "dress"
