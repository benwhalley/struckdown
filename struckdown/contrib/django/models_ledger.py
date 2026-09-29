"""The usage ledger: every provider request as a row, grouped into spans.

``LLMCall`` is written by :mod:`struckdown.contrib.django.ledger` from the
usage records struckdown emits; nothing else should write it. A row copies
the model's name, prices and residency as they were when the call was made,
so a repriced or deleted ``AvailableModel`` leaves history alone. The link
back to the ``AvailableModel`` row is for filtering, never a source of facts.

``LLMSpan`` is what a call belongs to: the HTTP request or Celery task it ran
in (a root span the middleware and task hooks open), or a named block a
caller opened with :func:`~struckdown.contrib.django.spans.llm_span`. Spans
nest, carry the user, and may point at the domain record that already holds
the text of the exchange, which is why the ledger stores none itself.

``LLMCallPayload`` is the exception: the bodies of a call, kept only when a
span or setting asked for them, and pruned on a short window.
"""

import uuid

from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.db.models import F
from django.utils import timezone


class LLMSpan(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=160,
        db_index=True,
        help_text="What this span is: 'http:tower:module_detail', 'celery:hub_tagging.autotag', "
        "'hub_assistant.turn'.",
    )
    parent = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="llm_spans",
    )
    # the domain record that holds the text of this exchange, if there is one
    content_type = models.ForeignKey(
        ContentType, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    object_id = models.CharField(max_length=64, blank=True)
    obj = GenericForeignKey("content_type", "object_id")
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    attributes = models.JSONField(default=dict, blank=True)
    capture_payloads = models.BooleanField(default=False)

    class Meta:
        db_table = "llm_span"
        ordering = ["-started_at"]
        indexes = [models.Index(fields=["name", "started_at"])]

    def __str__(self):
        return f"{self.name} @ {self.started_at:%Y-%m-%d %H:%M}"

    @property
    def root_name(self) -> str:
        span = self
        while span.parent_id:
            span = span.parent
        return span.name


class LLMCall(models.Model):
    class Kind(models.TextChoices):
        CHAT = "chat", "Chat"
        EMBEDDING = "embedding", "Embedding"
        TRANSCRIPTION = "transcription", "Transcription"

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    span = models.ForeignKey(
        LLMSpan, null=True, blank=True, on_delete=models.SET_NULL, related_name="calls"
    )
    # the span's root, denormalised so the costs page groups without walking parents
    root_name = models.CharField(max_length=160, blank=True, db_index=True)
    slot = models.CharField(max_length=120, blank=True)
    kind = models.CharField(max_length=20, choices=Kind.choices)

    # -- the model, as it was when called --
    model_name = models.CharField(max_length=200, db_index=True)
    provider = models.CharField(max_length=60, blank=True)
    base_url_host = models.CharField(max_length=200, blank=True)
    data_residency = models.CharField(max_length=10, blank=True)
    available_model = models.ForeignKey(
        "sd_models.AvailableModel",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="calls",
        help_text="A link for filtering. The columns on this row are the record.",
    )

    # -- prices used, USD per Mtok --
    input_price = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    output_price = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    cache_read_price = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    cache_write_price = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    price_source = models.CharField(max_length=20, blank=True)

    # -- tokens --
    input_tokens = models.PositiveIntegerField(default=0, help_text="All input, cached included.")
    cache_read_tokens = models.PositiveIntegerField(default=0)
    cache_write_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    reasoning_tokens = models.PositiveIntegerField(default=0, help_text="A subset of output.")
    audio_seconds = models.FloatField(null=True, blank=True)

    # -- cost, USD. Null means unknown, never 0. Set together or not at all. --
    input_cost = models.DecimalField(max_digits=14, decimal_places=8, null=True, blank=True)
    output_cost = models.DecimalField(max_digits=14, decimal_places=8, null=True, blank=True)
    total_cost = models.GeneratedField(
        expression=F("input_cost") + F("output_cost"),
        output_field=models.DecimalField(max_digits=14, decimal_places=8, null=True),
        db_persist=True,
    )

    # -- outcome --
    cache_hit = models.BooleanField(default=False, help_text="Served from struckdown's cache.")
    ok = models.BooleanField(default=True)
    error_class = models.CharField(max_length=120, blank=True)
    provider_request_id = models.CharField(max_length=120, blank=True)
    finish_reason = models.CharField(max_length=40, blank=True)

    class Meta:
        db_table = "llm_call"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["span", "created_at"]),
            models.Index(fields=["root_name", "created_at"]),
            models.Index(fields=["kind", "created_at"]),
        ]

    def __str__(self):
        cost = f"${self.total_cost:.4f}" if self.total_cost is not None else "unpriced"
        return f"{self.kind} {self.model_name} {cost} @ {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def unpriced(self) -> bool:
        return self.input_cost is None


class LLMCallPayload(models.Model):
    call = models.OneToOneField(LLMCall, on_delete=models.CASCADE, related_name="payload")
    request = models.JSONField(null=True, blank=True)
    response = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "llm_call_payload"

    def __str__(self):
        return f"payload of {self.call_id}"


class LLMCosts(LLMCall):
    """Proxy of LLMCall whose admin changelist is the costs page.

    Exists to give the page a natural admin URL (``/admin/sd_models/llmcosts/``),
    an index entry and a permission; it stores nothing of its own.
    """

    class Meta:
        proxy = True
        verbose_name = "LLM costs"
        verbose_name_plural = "LLM costs"
