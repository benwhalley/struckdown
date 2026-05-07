"""Speech-to-text helpers analogous to ``get_embedding`` for embedding models.

Public API: :func:`transcribe`, :func:`transcribe_async`, :class:`TranscriptionResult`,
:func:`set_audio_pricing`. Dispatch between the OpenAI-compatible and Azure
clients is explicit, driven by the ``provider:model`` prefix on ``model``
(``azure:whisper`` -> ``AzureOpenAI``; anything else -> ``OpenAI``). No URL sniffing.
"""

from __future__ import annotations

import asyncio
import io
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .audio_probe import (
    AudioSource,
    AudioValidation,
    validate_audio_for_transcription,
)
from .llm import LLMCredentials, parse_model_id

DEFAULT_AZURE_API_VERSION = "2024-06-01"

_audio_pricing: ContextVar[Optional[float]] = ContextVar("audio_pricing", default=None)


def set_audio_pricing(cost_per_minute: Optional[float]) -> None:
    """Set per-minute USD cost for the current context. Mirrors :func:`set_model_pricing`."""
    if cost_per_minute is not None:
        _audio_pricing.set(float(cost_per_minute))
    else:
        _audio_pricing.set(None)


@dataclass
class TranscriptionResult:
    text: str
    duration_s: Optional[float]
    estimated_duration_s: Optional[float]
    language: Optional[str]
    model: str
    cost: Optional[float]
    raw: Any = field(repr=False, default=None)


def _coerce_audio_file(
    source: AudioSource,
    validation: AudioValidation,
) -> tuple[io.BytesIO, str]:
    """Materialise to BytesIO with a sensible filename hint for the SDK upload."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = f"audio.{validation.detected_format or 'bin'}"
    elif isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        name = path.name
    else:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        data = source.read()
        name = getattr(source, "name", None) or f"audio.{validation.detected_format or 'bin'}"
        if isinstance(name, str):
            name = Path(name).name
        else:
            name = f"audio.{validation.detected_format or 'bin'}"
    buf = io.BytesIO(data)
    buf.name = name
    return buf, name


def _build_client(model: str, credentials: LLMCredentials):
    """Return (sdk_client, model_arg_for_api). Dispatch is explicit on the prefix."""
    provider_prefix, bare_name = parse_model_id(model)

    if provider_prefix == "azure":
        from openai import AzureOpenAI

        if not credentials.base_url:
            raise ValueError(
                "azure:* models require credentials.base_url to point at the "
                "Azure resource (e.g. https://<resource>.openai.azure.com/)"
            )
        client = AzureOpenAI(
            api_key=credentials.api_key,
            api_version=DEFAULT_AZURE_API_VERSION,
            azure_endpoint=credentials.base_url,
        )
        # Azure: the bare name is the deployment name.
        return client, bare_name

    from openai import OpenAI

    client = OpenAI(api_key=credentials.api_key, base_url=credentials.base_url or None)
    # OpenAI / proxy: pass through whatever model id the caller gave us
    # (strip the prefix only if one was set; bare names pass straight through).
    return client, bare_name if provider_prefix and provider_prefix != "openai-compatible" else model


def transcribe(
    audio: AudioSource,
    model: str = "whisper-1",
    credentials: Optional[LLMCredentials] = None,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "verbose_json",
) -> TranscriptionResult:
    """Synchronously transcribe an audio file.

    ``model`` follows struckdown's ``provider:bare_name`` convention. Use
    ``azure:<deployment>`` for Azure OpenAI deployments; anything else routes
    through the plain OpenAI client (works for OpenAI direct and OpenAI-compatible
    proxies like litellm).
    """
    if credentials is None:
        raise ValueError(
            "credentials are required: pass credentials=LLMCredentials(...) or "
            "use AvailableModel.get_llm_and_credentials()"
        )

    validation = validate_audio_for_transcription(audio)
    file_obj, _name = _coerce_audio_file(audio, validation)
    client, model_arg = _build_client(model, credentials)

    kwargs: dict[str, Any] = {
        "model": model_arg,
        "file": file_obj,
        "response_format": response_format,
    }
    if language is not None:
        kwargs["language"] = language
    if prompt is not None:
        kwargs["prompt"] = prompt

    raw = client.audio.transcriptions.create(**kwargs)

    text = getattr(raw, "text", None) or (raw if isinstance(raw, str) else str(raw))
    duration = getattr(raw, "duration", None)
    if duration is not None:
        duration = float(duration)
    detected_language = getattr(raw, "language", None)

    cost_per_minute = _audio_pricing.get()
    cost = None
    if cost_per_minute is not None and duration is not None:
        cost = cost_per_minute * duration / 60.0

    return TranscriptionResult(
        text=str(text),
        duration_s=duration,
        estimated_duration_s=validation.duration_s,
        language=detected_language,
        model=model,
        cost=cost,
        raw=raw,
    )


async def transcribe_async(
    audio: AudioSource,
    model: str = "whisper-1",
    credentials: Optional[LLMCredentials] = None,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "verbose_json",
) -> TranscriptionResult:
    """Async wrapper. The OpenAI SDK's audio endpoint is sync-only, so we offload
    the blocking call to a worker thread."""
    return await asyncio.to_thread(
        transcribe,
        audio,
        model,
        credentials,
        language=language,
        prompt=prompt,
        response_format=response_format,
    )
