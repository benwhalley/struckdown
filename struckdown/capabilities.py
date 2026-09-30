"""Does a model accept images? Looked up, not stored.

The same idea as :mod:`struckdown.pricing`: ask sources that already know,
rather than keep a flag on every model record. The first source with an
answer wins:

1. the LiteLLM proxy the request goes to (``{base_url}/model/info``), which
   describes the deployment actually called;
2. LiteLLM's public model registry (``supports_vision``);
3. OpenRouter's model list (``input_modalities`` includes ``"image"``).

``True``/``False`` if a source knows, ``None`` if none does -- the caller sends
anyway and lets the provider answer. Set ``STRUCKDOWN_CHECK_IMAGE_SUPPORT=0``
to skip the lookups.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import urllib.request
from functools import lru_cache
from typing import Optional

from .cache import memory as _memory

logger = logging.getLogger(__name__)

LITELLM_REGISTRY_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _get_json(url: str, api_key: Optional[str] = None, timeout: float = 10) -> dict:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _fetch_litellm_registry(_cache_day: int) -> dict:
    """{model: supports_vision} from LiteLLM's registry; ``_cache_day`` gives a 1-day TTL."""
    data = _get_json(LITELLM_REGISTRY_URL, timeout=20)
    return {k: v.get("supports_vision") for k, v in data.items() if isinstance(v, dict)}


def _fetch_openrouter_modalities(_cache_day: int) -> dict:
    """{model id: input modalities} from OpenRouter; 1-day TTL as above."""
    data = _get_json(OPENROUTER_MODELS_URL, timeout=15)
    return {
        m["id"]: (m.get("architecture") or {}).get("input_modalities") or []
        for m in data.get("data", [])
    }


_litellm_registry = _memory.cache(_fetch_litellm_registry)
_openrouter_modalities = _memory.cache(_fetch_openrouter_modalities)


def _bare(model_name: str) -> str:
    """``openai:gpt-4.1`` -> ``gpt-4.1``; names without a provider are unchanged."""
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


@lru_cache(maxsize=256)
def _from_proxy(base_url: str, api_key: Optional[str], model: str) -> Optional[bool]:
    for path in ("/model/info", "/v1/model/info"):
        try:
            data = _get_json(base_url.rstrip("/") + path, api_key)
        except Exception:
            continue
        for entry in data.get("data", []):
            if entry.get("model_name") == model:
                return (entry.get("model_info") or {}).get("supports_vision")
        return None
    return None


def _from_litellm_registry(model: str) -> Optional[bool]:
    registry = _litellm_registry(_cache_day=datetime.date.today().toordinal())
    for key in (model, f"openai/{model}", f"azure/{model}", f"anthropic/{model}", f"gemini/{model}"):
        if registry.get(key) is not None:
            return bool(registry[key])
    return None


def _from_openrouter(model: str) -> Optional[bool]:
    modalities = _openrouter_modalities(_cache_day=datetime.date.today().toordinal())
    matches = [m for mid, m in modalities.items() if mid == model or mid.endswith("/" + model)]
    return ("image" in matches[0]) if matches else None


def supports_images(model_name: Optional[str], credentials=None) -> tuple[Optional[bool], str]:
    """``(answer, source)``: whether ``model_name`` accepts image input, and who said so."""
    if not model_name or os.environ.get("STRUCKDOWN_CHECK_IMAGE_SUPPORT", "1") == "0":
        return None, "no lookup"
    model = _bare(model_name)
    base_url = getattr(credentials, "base_url", None)
    api_key = getattr(credentials, "api_key", None)
    lookups = [
        *([(f"the proxy at {base_url}", lambda: _from_proxy(base_url, api_key, model))] if base_url else []),
        ("LiteLLM's model registry", lambda: _from_litellm_registry(model)),
        ("OpenRouter's model list", lambda: _from_openrouter(model)),
    ]
    for source, lookup in lookups:
        try:
            answer = lookup()
        except Exception as e:  # a source being down must not stop the call
            logger.debug(f"image support lookup via {source} failed: {e}")
            continue
        if answer is not None:
            return answer, source
    return None, "no source"
