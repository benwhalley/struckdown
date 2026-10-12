"""Decision-model ("System One") backend.

Decision models (TypeSafe Jev, Microsoft-Decision-1) don't generate text. They take a
`state` plus typed questions and return calibrated probabilities over a fixed answer set:

- `noul`   -- yes/no, returns P(yes)
- `choice` -- one of N options, returns a probability per option
- `score`  -- ordered levels, returns the probability-weighted level index

A model is a decision model because its ModelSpec/LLM says `model_type="decision"`, never
because of its name. Both vendors speak the same `POST .../systemone` protocol; the model's
ordinary provider prefix only picks the endpoint and auth:

- `azure:<deployment>` -- Azure Foundry; base_url is the resource endpoint
  (`.../providers/microsoft/v1/systemone`, `api-key` header)
- `typesafe:jev-latest` -- api.typesafe.ai, Bearer auth
- no prefix, with base_url -- any gateway serving `/v1/systemone` with Bearer auth, e.g. a
  LiteLLM proxy (the model name is the proxy's alias)

Only slots whose return type carries a `decision_question` spec (pick, bool, decide, score,
halt) can be answered by a decision model. The spec is a dict: `type` (noul/choice/score),
`options` or `levels`, and optionally `field` (default "response") and `defaults` (other
fields of the return type to fill in; str.format templates over the answer, e.g. "{noul:.2f}").
"""

import logging
import os
import time
from typing import Any

import httpx
import tenacity

from .cache import memory
from .ledger import (
    CostBreakdown,
    ResponseStandIn,
    emit,
    now,
    record_from_response,
    slot_context,
)

logger = logging.getLogger(__name__)

# both vendors: input-only billing at $0.042 / Mtok, output free
INPUT_COST_PER_MTOK = 0.042
RETRY_ATTEMPTS = 6


def _connection(model_name: str, credentials) -> tuple[str, dict[str, str], str]:
    """(endpoint URL, auth headers, model name to send) for a decision model."""
    provider, bare = model_name.split(":", 1) if ":" in model_name else (None, model_name)
    if provider == "azure":
        base = credentials.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = credentials.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        suffix, headers = "/providers/microsoft/v1", {"api-key": key}
    elif provider == "typesafe":
        base = credentials.base_url or "https://api.typesafe.ai/v1"
        key = credentials.api_key or os.environ.get("TYPESAFE_API_KEY")
        suffix, headers = "/v1", {"Authorization": f"Bearer {key}"}
    elif provider is None:
        base = credentials.base_url
        suffix, headers = "/v1", {"Authorization": f"Bearer {credentials.api_key}"}
    else:
        raise ValueError(
            f"No decision-model endpoint for provider '{provider}' ({model_name}). Use "
            f"azure:<deployment>, typesafe:<model>, or an unprefixed name with a base_url "
            f"(a gateway such as a LiteLLM proxy)"
        )
    if not base:
        raise ValueError(f"decision model {model_name} needs a base_url")
    base = base.rstrip("/")
    base = base if base.endswith(suffix) else f"{base}{suffix}"
    return f"{base}/systemone", headers, bare


def messages_to_state(messages: list[dict[str, str]], *extra: str) -> str:
    """Flatten chat messages (plus any extra text) into one state string."""
    parts = [f"[{m['role']}]\n{m['content'].strip()}" for m in messages]
    return "\n\n".join(p for p in [*parts, *(e.strip() for e in extra)] if p)


def split_state_and_instructions(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Split a struckdown slot prompt into (state, instructions).

    The last paragraph of the final user message is taken as the question; everything
    before it (earlier turns, system text, preceding paragraphs) is the state being judged.
    """
    *earlier, last = messages
    paragraphs = [p.strip() for p in last["content"].strip().split("\n\n") if p.strip()]
    *context_paras, question = paragraphs or [""]
    state = messages_to_state(earlier, *context_paras) or question
    return state, question


def decision_spec(return_type) -> dict[str, Any]:
    spec = getattr(return_type, "decision_question", None)
    if spec is None:
        raise ValueError(
            f"a decision model can only answer pick, bool, decide, score or halt slots "
            f"(got {return_type.__name__})"
        )
    return spec


def build_question(spec: dict[str, Any], instructions: str) -> dict[str, Any]:
    kind = spec["type"]
    q = {"type": kind, "instructions": instructions}
    if kind == "choice":
        q["criteria"] = {opt: opt for opt in spec["options"]}
    elif kind == "score":
        q["criteria"] = list(spec["levels"])
    return q


def result_dict(spec: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """The return type's field values for a decision-model answer."""
    value = {
        "noul": lambda: answer["noul"] >= 0.5,
        "choice": lambda: answer["choice"],
        "score": lambda: answer["score"],
    }[spec["type"]]()
    defaults = {k: v.format(**answer) for k, v in spec.get("defaults", {}).items()}
    return {spec.get("field", "response"): value, **defaults}


def chat_answer(spec: dict[str, Any], value: Any) -> dict[str, Any]:
    """A decision-shaped (one-hot, uncalibrated) answer for a slot a chat model filled."""
    kind = spec["type"]
    if kind == "noul":
        answer = {"noul": None if value is None else float(bool(value))}
    elif kind == "choice":
        answer = {
            "choice": value,
            "probabilities": {o: float(o == value) for o in spec["options"]},
            "confidence": None,
        }
    else:
        legend = {str(i): lvl for i, lvl in enumerate(spec["levels"])}
        answer = {
            "score": value,
            "legend": legend,
            "probabilities": {i: float(value is not None and i == str(round(value))) for i in legend},
            "confidence": None,
        }
    return {"type": kind, "calibrated": False, **answer}


def _is_transient(exception: BaseException) -> bool:
    """Rate limits, server errors and dropped connections are worth retrying."""
    from .llm import TRANSIENT_STATUS_CODES

    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in TRANSIENT_STATUS_CODES
    return isinstance(exception, httpx.TransportError)


@memory.cache(ignore=["headers"])
@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_transient),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    before_sleep=lambda retry_state: logger.info(
        f"Decision model error ({retry_state.outcome.exception()}), retrying in "
        f"{retry_state.next_action.sleep:.1f}s "
        f"(attempt {retry_state.attempt_number}/{RETRY_ATTEMPTS})"
    ),
    reraise=True,
)
def _post_systemone(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ask(
    model_name: str,
    credentials,
    state: str,
    questions: dict[str, dict[str, Any]],
    timeout: float = 60.0,
) -> tuple[dict, dict, bool]:
    """POST one systemone request; return (payload, response body, was_cached)."""
    url, headers, model = _connection(model_name, credentials)
    payload = {"model": model, "state": state, "questions": questions}
    args = (url, payload, {**headers, "Accept": "application/json"}, timeout)
    was_cached = _post_systemone.check_call_in_cache(*args)
    return payload, _post_systemone(*args), was_cached


def completion_dict(
    messages: list[dict[str, str]], payload: dict, body: dict, answer: dict, input_tokens: int
) -> dict:
    """A completion dict in the shape StruckdownResult consumers expect."""
    cost = CostBreakdown(
        input_cost=input_tokens * INPUT_COST_PER_MTOK / 1e6,
        output_cost=0.0,
        input_price=INPUT_COST_PER_MTOK,
        output_price=0.0,
        source="decision_rate",
    )
    return {
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": 0,
            "total_tokens": input_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0},
        },
        "_hidden_params": {"response_cost": cost.total_cost},
        "_cost_breakdown": cost.to_dict(),
        "_request_messages": messages,
        "_thinking": None,
        "_decision": {"model": body.get("model"), "request": payload, "calibrated": True, **answer},
    }


def call_decision_model(
    messages: list[dict[str, str]], return_type, model_name: str, credentials
) -> tuple[dict, dict]:
    """Ask a decision model one question; return (res_dict, com_dict) like _call_llm_cached."""
    spec = decision_spec(return_type)
    state, instructions = split_state_and_instructions(messages)
    payload, body, _ = ask(model_name, credentials, state, {"q": build_question(spec, instructions)})
    answer = body["answers"]["q"]
    tokens = body.get("usage", {}).get("input_tokens", 0)
    return result_dict(spec, answer), completion_dict(messages, payload, body, answer, tokens)


def call_decision_batch(
    messages: list[dict[str, str]],
    preamble: str,
    slots: dict[str, tuple[Any, str]],
    model_name: str,
    credentials,
) -> tuple[dict[str, tuple[dict, dict]], bool]:
    """Ask several independent questions about the same state in one request.

    `slots` maps slot key -> (return_type, question text). The request's usage and cost are
    booked against the first slot so run totals stay right.
    Returns ({slot key: (res_dict, com_dict)}, was_cached).
    """
    specs = {key: decision_spec(rt) for key, (rt, _) in slots.items()}
    questions = {key: build_question(specs[key], text.strip()) for key, (_, text) in slots.items()}
    state = messages_to_state(messages, preamble)
    started_at, start = now(), time.monotonic()
    payload, body, was_cached = ask(model_name, credentials, state, questions)
    duration_ms = (time.monotonic() - start) * 1000
    tokens = body.get("usage", {}).get("input_tokens", 0)
    first = next(iter(slots))
    request_messages = [*messages, {"role": "user", "content": preamble}]
    results = {
        key: (
            result_dict(specs[key], body["answers"][key]),
            completion_dict(
                request_messages, payload, body, body["answers"][key], tokens if key == first else 0
            ),
        )
        for key in slots
    }
    # one ledger record for the request (single-question calls are recorded by structured_chat)
    first_com = results[first][1]
    with slot_context(",".join(slots)[:120]):
        emit(record_from_response(
            kind="chat",
            model_name=model_name,
            response=ResponseStandIn(first_com),
            cost=(CostBreakdown(input_cost=0.0, output_cost=0.0, source="cache") if was_cached
                  else CostBreakdown.from_dict(first_com["_cost_breakdown"])),
            credentials=credentials,
            started_at=started_at,
            duration_ms=duration_ms,
            cache_hit=was_cached,
        ))
    return results, was_cached
