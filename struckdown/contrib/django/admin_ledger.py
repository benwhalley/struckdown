"""Admin for the usage ledger: read-only changelists and the costs page.

Plain ``admin.ModelAdmin`` markup so the pages render under the stock admin
and under a themed one (unfold, grappelli) alike. A project on a custom admin
site or theme subclasses these and registers them itself; ``LLMCostsAdmin``
exposes ``extra_sections()`` for a host that wants to add its own tables
(spend recorded before the ledger existed, say).
"""

import json

from django.contrib import admin
from django.core.exceptions import PermissionDenied
from django.template.response import TemplateResponse
from django.utils.html import format_html

from . import costs
from .models import LLMCall, LLMCallPayload, LLMCosts, LLMSpan


class ReadOnlyAdmin(admin.ModelAdmin):
    """Rows are written by the ledger, never by hand. Deleting is allowed for cleanup."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields]


class LLMCallAdmin(ReadOnlyAdmin):
    list_display = [
        "created_at",
        "kind",
        "model_name",
        "root_name",
        "slot",
        "input_tokens",
        "cache_read_tokens",
        "output_tokens",
        "total_cost",
        "cache_hit",
        "ok",
        "duration_ms",
    ]
    list_filter = ["kind", "ok", "cache_hit", "price_source", "provider", "root_name", "model_name"]
    date_hierarchy = "created_at"
    search_fields = ["span__id", "model_name", "root_name", "slot", "error_class"]
    list_select_related = ["span"]
    raw_id_fields = ["span", "available_model"]


class LLMSpanAdmin(ReadOnlyAdmin):
    list_display = ["started_at", "name", "user", "parent", "call_count", "cost", "ended_at"]
    list_filter = ["name"]
    date_hierarchy = "started_at"
    search_fields = ["id", "name", "object_id"]
    raw_id_fields = ["parent", "user"]

    def get_queryset(self, request):
        from django.db.models import Count, Sum

        return (
            super()
            .get_queryset(request)
            .annotate(_calls=Count("calls"), _cost=Sum("calls__total_cost"))
        )

    @admin.display(description="Calls", ordering="_calls")
    def call_count(self, obj):
        return obj._calls

    @admin.display(description="Cost (USD)", ordering="_cost")
    def cost(self, obj):
        return f"{obj._cost:.4f}" if obj._cost is not None else "--"


class LLMCallPayloadAdmin(ReadOnlyAdmin):
    list_display = ["created_at", "call"]
    date_hierarchy = "created_at"
    raw_id_fields = ["call"]
    fields = ["call", "created_at", "request_pretty", "response_pretty"]

    def get_readonly_fields(self, request, obj=None):
        return self.fields

    @admin.display(description="Request")
    def request_pretty(self, obj):
        return format_html("<pre>{}</pre>", json.dumps(obj.request, indent=2, default=str))

    @admin.display(description="Response")
    def response_pretty(self, obj):
        return format_html("<pre>{}</pre>", json.dumps(obj.response, indent=2, default=str))


class LLMCostsAdmin(admin.ModelAdmin):
    """The costs page at ``/admin/sd_models/llmcosts/``.

    Overrides the changelist so the URL renders the summary, not a table.
    """

    template_name = "admin/sd_models/llmcosts/costs.html"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def extra_sections(self, request, days: int) -> list[dict]:
        """Tables a host adds below the ledger's own: ``[{"title", "columns", "rows", "note"}]``."""
        return []

    def changelist_view(self, request, extra_context=None):
        # the override skips ModelAdmin's own gate, so check it here
        if not self.has_view_permission(request):
            raise PermissionDenied
        try:
            days = int(request.GET.get("days", 30))
        except (TypeError, ValueError):
            days = 30
        if days not in [d for d, _ in costs.WINDOWS]:
            days = 30
        context = {
            **self.admin_site.each_context(request),
            "title": "LLM costs",
            "opts": self.model._meta,
            "days": days,
            "windows": costs.WINDOWS,
            "data": costs.dashboard(days),
            "extra_sections": self.extra_sections(request, days),
            "calls_url": f"{self.admin_site.name}:sd_models_llmcall_changelist",
            "spans_url": f"{self.admin_site.name}:sd_models_llmspan_changelist",
            **(extra_context or {}),
        }
        return TemplateResponse(request, self.template_name, context)


def register(site=admin.site, **overrides):
    """Register the ledger admins on ``site``, unless already there.

    ``overrides`` maps a model to the admin class to use instead, for a host
    that subclasses these (a themed admin base class, extra sections).
    """
    for model, cls in (
        (LLMCall, LLMCallAdmin),
        (LLMSpan, LLMSpanAdmin),
        (LLMCallPayload, LLMCallPayloadAdmin),
        (LLMCosts, LLMCostsAdmin),
    ):
        if not site.is_registered(model):
            site.register(model, overrides.get(model, cls))
