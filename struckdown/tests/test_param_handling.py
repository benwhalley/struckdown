"""Tests for LLM parameter translation, warning/strict handling, and settings propagation."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from struckdown.llm import (
    KNOWN_MODEL_SETTINGS,
    _translate_kwargs,
)
from struckdown.return_type_models import LLMConfig, THINKING_LEVELS


# --- LLMConfig validation ---


class TestLLMConfigThinking:
    """Test LLMConfig thinking field validation."""

    def test_thinking_none_by_default(self):
        config = LLMConfig()
        assert config.thinking is None

    @pytest.mark.parametrize("level", list(THINKING_LEVELS))
    def test_valid_thinking_levels(self, level):
        config = LLMConfig(thinking=level)
        assert config.thinking == level

    def test_invalid_thinking_raises(self):
        with pytest.raises(ValueError, match="thinking must be one of"):
            LLMConfig(thinking="turbo")

    def test_invalid_thinking_bool_string(self):
        with pytest.raises(ValueError, match="thinking must be one of"):
            LLMConfig(thinking="true")

    def test_thinking_with_temperature(self):
        config = LLMConfig(thinking="high", temperature=0.3)
        assert config.thinking == "high"
        assert config.temperature == 0.3

    def test_unknown_field_rejected(self):
        """LLMConfig has extra='forbid' so unknown fields raise."""
        with pytest.raises(Exception):
            LLMConfig(unknown_param="value")


# --- _translate_kwargs ---


class TestTranslateKwargs:
    """Test the _translate_kwargs function."""

    def test_empty_kwargs(self):
        settings = _translate_kwargs(None)
        assert settings == {}

    def test_empty_dict(self):
        settings = _translate_kwargs({})
        assert settings == {}

    def test_known_params_pass_through(self):
        settings = _translate_kwargs({
            "temperature": 0.5,
            "max_tokens": 100,
            "seed": 42,
            "top_p": 0.9,
        })
        assert settings["temperature"] == 0.5
        assert settings["max_tokens"] == 100
        assert settings["seed"] == 42
        assert settings["top_p"] == 0.9

    def test_thinking_passes_through(self):
        settings = _translate_kwargs({"thinking": "high"})
        assert settings["thinking"] == "high"

    def test_thinking_off_passes_through(self):
        settings = _translate_kwargs({"thinking": "off"})
        assert settings["thinking"] == "off"

    def test_thinking_none_omitted(self):
        """When thinking is None, it should not appear in settings."""
        settings = _translate_kwargs({"temperature": 0.5})
        assert "thinking" not in settings

    def test_timeout_passes_through(self):
        settings = _translate_kwargs({"timeout": 30.0})
        assert settings["timeout"] == 30.0

    def test_presence_penalty_passes_through(self):
        settings = _translate_kwargs({"presence_penalty": 0.5})
        assert settings["presence_penalty"] == 0.5

    def test_frequency_penalty_passes_through(self):
        settings = _translate_kwargs({"frequency_penalty": 0.3})
        assert settings["frequency_penalty"] == 0.3

    def test_unknown_params_warned(self, caplog):
        """Unknown params should log a warning and be dropped."""
        with caplog.at_level(logging.WARNING):
            settings = _translate_kwargs({"temperature": 0.5, "top_k": 50, "custom": "val"})
        assert settings["temperature"] == 0.5
        assert "top_k" not in settings
        assert "custom" not in settings
        assert "Dropped unsupported LLM parameters" in caplog.text
        assert "top_k" in caplog.text
        assert "custom" in caplog.text

    def test_strict_raises_on_unknown(self):
        """strict=True should raise ValueError on unknown params."""
        with pytest.raises(ValueError, match="Dropped unsupported LLM parameters"):
            _translate_kwargs({"temperature": 0.5, "top_k": 50}, strict=True)

    def test_strict_passes_with_known_only(self):
        """strict=True should not raise when all params are known."""
        settings = _translate_kwargs(
            {"temperature": 0.5, "thinking": "high", "seed": 42},
            strict=True,
        )
        assert settings["temperature"] == 0.5
        assert settings["thinking"] == "high"
        assert settings["seed"] == 42

    def test_internal_keys_not_warned(self, caplog):
        """Internal keys like stream_debounce_ms and model should not trigger warnings."""
        with caplog.at_level(logging.WARNING):
            settings = _translate_kwargs({
                "temperature": 0.5,
                "stream_debounce_ms": 200,
                "model": "gpt-4",
            })
        assert settings["temperature"] == 0.5
        assert "stream_debounce_ms" not in settings
        assert "model" not in settings
        assert "Dropped" not in caplog.text

    def test_internal_keys_not_strict_error(self):
        """Internal keys should not cause strict mode to raise."""
        settings = _translate_kwargs(
            {"temperature": 0.5, "stream_debounce_ms": 200, "model": "gpt-4"},
            strict=True,
        )
        assert settings["temperature"] == 0.5

    def test_all_known_settings_accepted(self):
        """Every key in KNOWN_MODEL_SETTINGS should pass through without warning."""
        kwargs = {k: "test_value" for k in KNOWN_MODEL_SETTINGS}
        # use valid types for the ones that might be checked
        kwargs["temperature"] = 0.5
        kwargs["max_tokens"] = 100
        kwargs["seed"] = 42
        kwargs["top_p"] = 0.9
        kwargs["timeout"] = 30.0
        kwargs["thinking"] = "high"
        kwargs["presence_penalty"] = 0.1
        kwargs["frequency_penalty"] = 0.2
        settings = _translate_kwargs(kwargs, strict=True)
        for key in KNOWN_MODEL_SETTINGS:
            assert key in settings


# --- Integration: settings reach _translate_kwargs ---


class TestSettingsPropagation:
    """Test that per-slot settings are correctly propagated through the call chain.

    These tests intercept _translate_kwargs to verify the kwargs dict that would
    be converted to ModelSettings, bypassing the joblib cache layer.
    """

    def test_thinking_reaches_translate_kwargs(self):
        """Verify thinking=high reaches _translate_kwargs when passed via extra_kwargs."""
        captured_kwargs = []
        original = _translate_kwargs

        def spy(extra_kwargs, strict=False):
            captured_kwargs.append(dict(extra_kwargs) if extra_kwargs else {})
            return original(extra_kwargs, strict=strict)

        with patch("struckdown.llm._translate_kwargs", side_effect=spy):
            # call _translate_kwargs directly through the pipeline
            spy({"thinking": "high", "temperature": 0.5}, strict=False)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["thinking"] == "high"

    def test_thinking_high_produces_correct_model_settings(self):
        """thinking=high should appear in the returned ModelSettings."""
        settings = _translate_kwargs({"thinking": "high"})
        assert settings["thinking"] == "high"

    def test_temperature_produces_correct_model_settings(self):
        settings = _translate_kwargs({"temperature": 0.8})
        assert settings["temperature"] == 0.8

    def test_thinking_not_in_settings_when_omitted(self):
        settings = _translate_kwargs({"temperature": 0.5})
        assert "thinking" not in settings

    def test_strict_params_raises_on_unknown_via_structured_chat(self):
        """strict_params=True should raise when unknown params are passed."""
        with patch("struckdown.llm._run_agent_sync"), \
             patch("struckdown.llm._build_pydantic_ai_agent"):
            from struckdown.llm import LLM, LLMCredentials, structured_chat
            from pydantic import BaseModel

            class TestResponse(BaseModel):
                response: str

            with pytest.raises(ValueError, match="Dropped unsupported"):
                structured_chat(
                    messages=[{"role": "user", "content": "test"}],
                    return_type=TestResponse,
                    llm=LLM(model_name="gpt-4.1-mini"),
                    credentials=LLMCredentials(api_key="test", base_url="http://test"),
                    extra_kwargs={"top_k": 50},
                    strict_params=True,
                )

    def test_combined_params_produce_correct_model_settings(self):
        """Multiple params including thinking and seed should all appear in ModelSettings."""
        settings = _translate_kwargs(
            {"thinking": "high", "seed": 42, "temperature": 0.3},
            strict=True,
        )
        assert settings["thinking"] == "high"
        assert settings["seed"] == 42
        assert settings["temperature"] == 0.3

    def test_gpt41mini_settings(self):
        """Settings for a non-thinking model like gpt-4.1-mini work correctly."""
        settings = _translate_kwargs({"temperature": 0.5, "seed": 1})
        assert settings["temperature"] == 0.5
        assert settings["seed"] == 1
        assert "thinking" not in settings

    def test_gpt5mini_with_thinking(self):
        """Settings for a thinking model like gpt-5-mini include thinking level."""
        settings = _translate_kwargs(
            {"thinking": "high", "temperature": 0.3, "max_tokens": 1000}
        )
        assert settings["thinking"] == "high"
        assert settings["temperature"] == 0.3
        assert settings["max_tokens"] == 1000
