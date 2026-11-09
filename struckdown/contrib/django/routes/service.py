"""Running a route: screen the inputs, run the prompt, shape the outputs.

Shared by the API view and the admin playground. Everything a call spends is
recorded in the ledger under one span pointing at the route, which is what
:meth:`PromptRoute.spent_today` sums for the daily budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from ..spans import llm_span
from . import schema as schemas


@dataclass
class RouteError(Exception):
    """A refusal with an HTTP status and a stable ``code`` callers can switch on."""

    status: int
    code: str
    message: str
    extra: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"error": self.code, "message": self.message, **self.extra}


def _model_pair(route):
    from struckdown.llm import LLMCredentials

    llm, credentials = route.model.get_llm_and_credentials()
    if route.credential_id:
        cred = route.credential
        credentials = LLMCredentials(api_key=cred.api_key, base_url=cred.base_url or None)
    return llm, credentials


def _complete(route, context: dict):
    import struckdown as sd

    llm, credentials = _model_pair(route)
    return sd.complete(route.prompt, context=context, model=llm, credentials=credentials)


def _cost(results) -> dict:
    return {
        "usd": round(sum(r.total_cost for r in results), 6),
        # true when some call's price was not known, so "usd" is a floor
        "unknown": any(r.has_unknown_costs for r in results),
    }


def outputs(route, result) -> dict:
    data = {k: to_jsonable_python(v.output) for k, v in result.results.items()}
    if not route.output_schema:
        return data
    model = schemas.to_model(route.output_schema, max_chars=10**9, name="Outputs")
    try:
        shaped = model(**{k: data.get(k) for k in model.model_fields})
    except ValidationError as exc:
        raise RouteError(
            502,
            "output_invalid",
            "The model's answer did not match the route's output schema.",
            {"detail": exc.errors(include_url=False, include_context=False, include_input=False)},
        ) from exc
    return shaped.model_dump(mode="json")


def screened(route, context: dict) -> tuple[list[str], Any]:
    """Names of the screen's flags that came back true, and its result."""
    screen = route.screen
    inputs = set(screen.analysis()["inputs"])
    result = _complete(screen, {k: v for k, v in context.items() if k in inputs})
    return [k for k, v in result.results.items() if v.output is True], result


def call(route, context: dict, *, user=None, caller: str | None = None) -> dict:
    """Run ``route`` on validated ``context``. Returns ``{"data", "cost"}``;
    raises :class:`RouteError` if the screen stops it."""
    results = []
    with llm_span("sd_routes.call", user=user, obj=route, route=route.slug, caller=caller or ""):
        if route.screen_id:
            flags, screen_result = screened(route, context)
            results.append(screen_result)
            if flags:
                raise RouteError(
                    422,
                    "input_rejected",
                    "The input was not processed.",
                    {"reasons": flags, "cost": _cost(results)},
                )
        result = _complete(route, context)
        results.append(result)
    return {"data": outputs(route, result), "cost": _cost(results)}


def validate_inputs(route, data: dict, files: dict) -> dict:
    """The template context for ``data`` (text) and ``files`` (uploads), or a
    400 :class:`RouteError` saying what is wrong."""
    from struckdown import attach

    from .uploads import NotAnImage, read_image

    input_schema = route.effective_input_schema()
    image_names = schemas.image_fields(input_schema)
    model = schemas.to_model(input_schema, max_chars=route.max_chars)
    try:
        inputs = model(**data).model_dump()
    except ValidationError as exc:
        raise RouteError(
            400,
            "invalid_input",
            "The input did not match what this route accepts.",
            {"detail": exc.errors(include_url=False, include_context=False, include_input=False)},
        ) from exc

    unexpected = set(files) - image_names
    if unexpected:
        raise RouteError(400, "invalid_input", f"Unexpected file field(s): {sorted(unexpected)}.")
    required = set(input_schema.get("required") or [])
    missing = (image_names & required) - set(files)
    if missing:
        raise RouteError(400, "invalid_input", f"Missing image(s): {sorted(missing)}.")

    images = {}
    for name, upload in files.items():
        try:
            images[name] = attach(read_image(upload))
        except NotAnImage as exc:
            raise RouteError(400, "invalid_image", f"{name}: {exc}.") from exc
    # an optional input left out reads as empty, not as the word "None"
    return {**{k: ("" if v is None else v) for k, v in inputs.items()}, **images}
