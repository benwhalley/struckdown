"""Per-slot model routing.

A slot can name its own model with `model=<alias>`, and `[[decide:...]]` or `mode=decision`
asks for a decision model. Names are matched against the models passed to complete()
(`registry=`), which are held in a context var for the duration of the run.
"""

import logging
from contextvars import ContextVar

from .llm import LLM, LLMCredentials
from .model_spec import ModelRegistry, ModelSpec

logger = logging.getLogger(__name__)

_active_registry: ContextVar[ModelRegistry | None] = ContextVar(
    "struckdown_active_registry", default=None
)


def set_active_registry(registry: ModelRegistry | None) -> None:
    _active_registry.set(registry)


def wants_decision(slot_info) -> bool:
    return (
        slot_info.action_type == "decide"
        or (slot_info.llm_kwargs or {}).get("mode") == "decision"
    )


def _slot_label(slot_info) -> str:
    return f"[[{slot_info.action_type}:{slot_info.key}]]"


def _match(registry: ModelRegistry | None, name: str, slot_info) -> ModelSpec:
    # no models passed: take the name as given, on the run's credentials (e.g. a proxy alias)
    if registry is None:
        return ModelSpec(model_name=name)
    spec, exact = registry.match(name)
    if not exact:
        logger.warning(
            f"\n{'!' * 72}\nFUZZY MODEL MATCH: {_slot_label(slot_info)} asked for "
            f"model={name!r}, using {spec.model_name!r}. Use the exact alias or name to "
            f"silence this.\n{'!' * 72}"
        )
    return spec


def resolve_slot_model(
    slot_info, llm: LLM, credentials: LLMCredentials
) -> tuple[LLM, LLMCredentials]:
    """The model a slot runs on: its model= option, a decision model, or the run default."""
    name = (slot_info.llm_kwargs or {}).get("model")
    decision = wants_decision(slot_info)
    registry = _active_registry.get()

    if not name and not (decision and llm.model_type != "decision"):
        return llm, credentials

    if name:
        spec = _match(registry, name, slot_info)
    elif registry is not None:
        spec = registry.resolve_decision()
    else:
        raise ValueError(
            f"{_slot_label(slot_info)} needs a decision model, but none was passed to "
            f"complete(). Pass registry=[ModelSpec(..., model_type='decision'), ...] "
            f"or give the slot model=<decision model>"
        )

    if decision and spec.model_type != "decision":
        raise ValueError(
            f"{_slot_label(slot_info)} needs a decision model, but {spec.model_name} isn't one"
        )

    creds = spec.as_credentials() if (spec.api_key or spec.base_url) else credentials
    return spec.as_llm(), creds
