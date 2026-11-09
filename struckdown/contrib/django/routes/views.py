"""``POST /api/<slug>/``: run a route for a caller holding one of its tokens.

Checks, in order, each a JSON ``{"error": <code>, "message": ...}``:

- 404 ``not_found`` -- no such route
- 401 ``unauthorised`` -- no token, or not an active token for this route
- 403 ``origin_not_allowed`` -- a browser call from a page not on the route's list
- 403 ``closed`` -- inactive, or outside its open window
- 429 ``rate_limited`` -- this token's calls this minute are used up
- 429 ``budget_exhausted`` -- the route has spent its daily budget
- 429 ``busy`` -- too many calls to this route are running at once
- 413 ``too_large`` -- the request body is bigger than the route could accept
- 400 ``invalid_input`` / ``invalid_image`` / ``invalid_json``
- 422 ``input_rejected`` -- the screen flagged the input (``reasons`` names the flags)
- 502 ``output_invalid`` / ``model_error``

A success is ``{"data": {<slot>: <value>...}, "cost": {"usd": ..., "unknown": ...}}``.

The token goes in a header (``Authorization: Bearer <token>``), never the URL,
so it stays out of server logs. Auth is by header rather than cookie, so the
view is CSRF-exempt and a browser page on another site can call it when the
route lists that page's origin.
"""

from __future__ import annotations

import json
import logging
import time

from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import RequestDataTooBig
from django.http import JsonResponse
from django.utils import timezone
from django.utils.module_loading import import_string
from django.views.decorators.csrf import csrf_exempt

from .models import PromptRoute, RouteToken
from .service import RouteError, call, validate_inputs
from .uploads import MemoryOnlyUploadHandler

logger = logging.getLogger(__name__)

TOKEN_HEADERS = ("HTTP_AUTHORIZATION", "HTTP_X_ROUTE_TOKEN")
# room for the text fields alongside the images
TEXT_ALLOWANCE = 1024 * 1024


def _token(request):
    for header in TOKEN_HEADERS:
        raw = request.META.get(header, "").strip()
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        if raw:
            return raw if len(raw) <= 64 else None
    return None


def _cors(response, route, origin):
    if origin is None:
        return response
    allowed = route.origins() if route else []
    if not allowed:
        response["Access-Control-Allow-Origin"] = "*"
    elif origin in allowed:
        response["Access-Control-Allow-Origin"] = origin
        response["Vary"] = "Origin"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Route-Token"
    response["Access-Control-Max-Age"] = "600"
    return response


def _setting(name: str, default):
    return getattr(settings, f"STRUCKDOWN_ROUTES_{name}", default)


def _client_ip(request) -> str:
    """``STRUCKDOWN_ROUTES_CLIENT_IP`` (a dotted path to ``f(request) -> str``)
    when the site is behind a proxy; otherwise the connecting address."""
    path = _setting("CLIENT_IP", None)
    return import_string(path)(request) if path else request.META.get("REMOTE_ADDR", "")


def _count(key: str, limit: int) -> bool:
    """Count one hit this minute against ``key``; True when over ``limit``."""
    key = f"{key}:{int(time.time() // 60)}"
    if cache.add(key, 1, timeout=90):
        return limit < 1
    try:
        return cache.incr(key) > limit
    except ValueError:
        cache.add(key, 1, timeout=90)
        return False


class _InFlight:
    """At most ``STRUCKDOWN_ROUTES_MAX_IN_FLIGHT`` calls to one route at once.

    The budget is checked against the ledger, which only hears of a call when
    it ends, so every call already running passes the check; capping how many
    can run bounds how far a day's spend can pass the budget."""

    def __init__(self, route):
        self.key = f"sd_routes:inflight:{route.pk}"
        self.limit = _setting("MAX_IN_FLIGHT", 4)

    def __enter__(self):
        cache.add(self.key, 0, timeout=300)
        try:
            running = cache.incr(self.key)
        except ValueError:
            cache.add(self.key, 1, timeout=300)
            running = 1
        if running > self.limit:
            self.__exit__()
            raise RouteError(429, "busy", "Too many calls to this route are running. Try again.")
        return self

    def __exit__(self, *exc):
        try:
            cache.decr(self.key)
        except ValueError:
            pass


def _max_body(route) -> int:
    images = route.max_images * route.max_image_bytes if route.allow_images else 0
    return images + route.max_chars * 8 + TEXT_ALLOWANCE


def _payload(request, route) -> tuple[dict, dict]:
    """``(text fields, files)`` from a JSON or form body."""
    content_type = request.content_type or ""
    if content_type == "application/json":
        try:
            data = json.loads(request.body or b"{}")
        except (ValueError, RecursionError) as exc:
            raise RouteError(400, "invalid_json", "The body is not valid JSON.") from exc
        if not isinstance(data, dict):
            raise RouteError(400, "invalid_json", "The body must be a JSON object.")
        return data, {}

    handler = MemoryOnlyUploadHandler(
        request,
        max_bytes=route.max_image_bytes,
        max_files=route.max_images if route.allow_images else 0,
    )
    request.upload_handlers = [handler]
    data = {k: (v if len(v) > 1 else v[0]) for k, v in request.POST.lists()}
    files = {k: request.FILES[k] for k in request.FILES}
    if handler.refused:
        detail = "; ".join(f"{k}: {why}" for k, why in handler.refused.items())
        raise RouteError(400, "invalid_image", f"Image refused ({detail}).")
    return data, files


@csrf_exempt
def route_api(request, slug):
    origin = request.META.get("HTTP_ORIGIN")
    route = PromptRoute.objects.select_related("model", "credential", "screen").filter(
        slug=slug
    ).first()

    def reply(body, status=200):
        return _cors(JsonResponse(body, status=status), route, origin)

    if request.method == "OPTIONS":
        return reply({}, 204 if route else 404)
    if request.method != "POST":
        return reply({"error": "method_not_allowed", "message": "POST only."}, 405)

    try:
        # before the token lookup, so it also slows guessing
        if _count(f"sd_routes:ip:{_client_ip(request)}", _setting("PER_IP_PER_MINUTE", 60)):
            raise RouteError(429, "rate_limited", "Too many calls from this address.")
        if route is None:
            raise RouteError(404, "not_found", "No such route.")
        token_value = _token(request)
        token = (
            RouteToken.objects.filter(route=route, token=token_value, is_active=True).first()
            if token_value
            else None
        )
        if token is None:
            raise RouteError(401, "unauthorised", "Send an active token for this route.")
        if origin and route.origins() and origin not in route.origins():
            raise RouteError(403, "origin_not_allowed", "Calls from this page are not allowed.")
        if not route.is_open():
            raise RouteError(403, "closed", "This route is not open.")
        if _count(f"sd_routes:rate:{token.pk}", route.per_minute):
            raise RouteError(429, "rate_limited", "Too many calls this minute.")
        if route.over_budget():
            raise RouteError(429, "budget_exhausted", "Today's budget for this route is spent.")
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError as exc:
            raise RouteError(400, "invalid_request", "Bad Content-Length.") from exc
        if length > _max_body(route):
            raise RouteError(413, "too_large", "The request is too large.")

        try:
            data, files = _payload(request, route)
        except RequestDataTooBig as exc:
            raise RouteError(413, "too_large", "The request is too large.") from exc
        context = validate_inputs(route, data, files)
        RouteToken.objects.filter(pk=token.pk).update(last_used_at=timezone.now())
        try:
            with _InFlight(route):
                return reply(call(route, context, caller=token.name))
        except RouteError:
            raise
        except Exception as exc:
            # no traceback: an error tracker would attach the frame's locals,
            # which hold the token and the caller's input
            logger.error("route %s failed: %s", route.slug, type(exc).__name__)
            raise RouteError(502, "model_error", "The model call failed.") from exc
    except RouteError as exc:
        return reply(exc.as_json(), exc.status)
