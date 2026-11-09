"""A struckdown prompt served at ``/api/<slug>/``, and the tokens that may call it."""

from __future__ import annotations

import re
from datetime import timedelta
from decimal import Decimal

import shortuuid
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Sum
from django.utils import timezone

from ..models import AvailableModel, Credential
from . import schema as schemas

# Jinja that reads files from the server. A route's prompt is written by staff
# but runs on every request, so it may only see what the request sends. (The
# sandbox has no loader, so these fail anyway; refusing them says so on save.)
_FILE_TAGS = re.compile(r"{%[-+]?\s*(include|import|from|extends)\b")

# The only actions a route may run: none of them reads a file or the network.
# Checked on save from the parse tree, and enforced again at run time
# (struckdown.restrict_actions), so a prompt cannot reach another by any spelling.
ROUTE_ACTIONS = frozenset({"set", "timestamp", "markdownify"})


def actions_used(prompt: str) -> set[str]:
    """Every action the prompt calls, as the parser reads it."""
    from struckdown.parsing import parse_syntax

    return {
        part.action_type
        for segment in parse_syntax(prompt)
        for part in segment.values()
        if getattr(part, "is_function", False)
    }


class PromptRoute(models.Model):
    slug = models.SlugField(
        unique=True, help_text="The last part of the endpoint's URL; a POST there runs this prompt."
    )
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, help_text="For staff; not shown to callers.")
    prompt = models.TextField(
        help_text="Struckdown. {{ variables }} are the inputs; [[slots]] are the outputs."
    )

    model = models.ForeignKey(
        AvailableModel,
        on_delete=models.PROTECT,
        related_name="prompt_routes",
        limit_choices_to={"model_type__in": ["llm", "decision"], "is_active": True},
    )
    credential = models.ForeignKey(
        Credential,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="prompt_routes",
        help_text="Leave blank to use the model's own credential.",
    )

    is_active = models.BooleanField(default=True)
    opens_at = models.DateTimeField(null=True, blank=True, help_text="Blank: open now.")
    closes_at = models.DateTimeField(null=True, blank=True, help_text="Blank: never closes.")

    daily_budget = models.DecimalField(
        "Daily budget (USD)",
        max_digits=8,
        decimal_places=2,
        default=Decimal("5.00"),
        help_text="Calls stop for the day once this much has been spent (UTC day).",
    )
    per_minute = models.PositiveIntegerField(
        default=20, help_text="Calls a minute allowed for each token."
    )
    max_input_tokens = models.PositiveIntegerField(
        default=10_000,
        help_text="Cap on each text input, in tokens (counted as 4 characters each).",
    )

    input_schema = models.JSONField(
        null=True,
        blank=True,
        help_text=(
            "Optional JSON Schema for the inputs. Blank: every {{ variable }} is a "
            "required string. A property with \"format\": \"image\" is an uploaded image."
        ),
    )
    output_schema = models.JSONField(
        null=True,
        blank=True,
        help_text="Optional JSON Schema choosing and typing the slots returned. Blank: all.",
    )

    allow_images = models.BooleanField(
        default=False,
        help_text=(
            "Accept image uploads for inputs marked \"format\": \"image\". Images are "
            "held in memory, re-encoded and sent to the provider; never saved."
        ),
    )
    max_image_bytes = models.PositiveIntegerField(default=5 * 1024 * 1024)
    max_images = models.PositiveSmallIntegerField(default=4)

    allowed_origins = models.TextField(
        blank=True,
        help_text=(
            "Web pages that may call this from a browser, one origin per line "
            "(https://example.org). Blank: any."
        ),
    )

    screen = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="screened_routes",
        help_text=(
            "A route run on the inputs first. Its slots must all be [[bool:...]]; if any "
            "comes back true (e.g. distress, injection) the request stops with that name."
        ),
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["slug"]
        verbose_name = "API route"

    def __str__(self):
        return self.endpoint_path()

    def endpoint_path(self) -> str:
        """Where the project mounted the routes, e.g. ``/api/sd/<slug>/``."""
        from django.urls import reverse

        return reverse("sd_routes:call", args=[self.slug])

    @property
    def max_chars(self) -> int:
        return self.max_input_tokens * schemas.CHARS_PER_TOKEN

    # -- what the prompt says -----------------------------------------------

    def analysis(self) -> dict:
        return schemas.analyse(self.prompt)

    def effective_input_schema(self) -> dict:
        return self.input_schema or schemas.derived_input_schema(
            self.prompt, self.max_input_tokens
        )

    def origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.allowed_origins.splitlines() if o.strip()]

    def clean(self):
        errors = {}
        found = self.analysis()
        if found["error"]:
            errors["prompt"] = f"The prompt does not parse: {found['error']}"
        if _FILE_TAGS.search(self.prompt or ""):
            errors["prompt"] = "include, import and extends are not allowed in a route's prompt."
        elif not found["error"]:
            blocked = actions_used(self.prompt) - ROUTE_ACTIONS
            if blocked:
                errors["prompt"] = (
                    f"Actions not allowed in a route: {', '.join(sorted(blocked))} "
                    f"(allowed: {', '.join(sorted(ROUTE_ACTIONS))})."
                )
        # an unpriced model records no cost, so the daily budget could never stop it
        if self.model_id and self.model.stored_pricing() is None:
            errors["model"] = "This model has no prices, so the daily budget could not apply."
        for field in ("input_schema", "output_schema"):
            try:
                schemas.check_schema(
                    getattr(self, field),
                    max_chars=self.max_chars,
                    inputs=field == "input_schema",
                )
            except schemas.SchemaError as exc:
                errors[field] = str(exc)
        if schemas.image_fields(self.input_schema) and not self.allow_images:
            errors["allow_images"] = "The input schema has image inputs; allow images to use it."
        if self.opens_at and self.closes_at and self.closes_at <= self.opens_at:
            errors["closes_at"] = "Must be after it opens."
        if self.screen_id:
            errors.update(self._screen_errors(self, self.screen))
        # a screen's rules are checked when it is saved too, so editing the
        # screen itself cannot quietly stop it screening
        if self.pk:
            for screened in self.screened_routes.all():
                for message in self._screen_errors(screened, self).values():
                    errors.setdefault("prompt", f"Used as the screen for {screened}: {message}")
        if errors:
            raise ValidationError(errors)

    @staticmethod
    def _screen_errors(route, screen) -> dict:
        if screen.pk == route.pk:
            return {"screen": "A route cannot screen itself."}
        if screen.screen_id:
            return {"screen": "A screening route cannot have a screen of its own."}
        if screen.model.model_type != AvailableModel.ModelType.DECISION:
            return {"screen": "A screening route must use a decision model."}
        if screen.model.stored_pricing() is None:
            return {"screen": "The screen's model has no prices, so its cost could not count."}
        slots = screen.analysis()["slots"]
        if not slots or any(s["type"] not in ("bool", "boolean") for s in slots):
            return {"screen": "A screening route's slots must all be [[bool:...]]."}
        missing = set(screen.analysis()["inputs"]) - set(route.analysis()["inputs"])
        if missing:
            return {"screen": f"The screen reads inputs this route has not: {sorted(missing)}."}
        return {}

    # -- whether it can be called now --------------------------------------

    def is_open(self, now=None) -> bool:
        now = now or timezone.now()
        return (
            self.is_active
            and (self.opens_at is None or self.opens_at <= now)
            and (self.closes_at is None or now < self.closes_at)
        )

    def spent_today(self) -> Decimal:
        """USD recorded in the ledger against this route since midnight UTC,
        including what its screen spent on its behalf."""
        from django.contrib.contenttypes.models import ContentType

        from ..models_ledger import LLMCall

        midnight = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        total = LLMCall.objects.filter(
            span__content_type=ContentType.objects.get_for_model(PromptRoute),
            span__object_id=str(self.pk),
            created_at__gte=midnight,
            created_at__lt=midnight + timedelta(days=1),
        ).aggregate(total=Sum("total_cost"))["total"]
        return total or Decimal(0)

    def over_budget(self) -> bool:
        return self.spent_today() >= self.daily_budget


def new_token() -> str:
    """22 characters from shortuuid's unambiguous alphabet (about 128 bits),
    with no hyphens, so it survives a double-click select and a copy-paste."""
    return shortuuid.uuid()


class RouteToken(models.Model):
    """A key that may call one route. Give each caller (a class, an
    experiment, an app) its own, so one can be revoked without the others."""

    route = models.ForeignKey(PromptRoute, on_delete=models.CASCADE, related_name="tokens")
    name = models.CharField(max_length=120, help_text="Who has it, e.g. 'PSYC611 card sort'.")
    token = models.CharField(max_length=40, default=new_token, unique=True, editable=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True, editable=False)

    class Meta:
        ordering = ["route", "name"]

    def __str__(self):
        return f"{self.route.slug}: {self.name}"

    def rotate(self):
        self.token = new_token()
