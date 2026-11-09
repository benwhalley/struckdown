"""Admin for API routes: the prompt editor, schema helpers, tokens and a playground.

Registered on the default admin site unless the project has registered its own
(as with the sd_models admins).
"""

from __future__ import annotations

import json

from django import forms
from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html, format_html_join

from . import examples
from . import schema as schemas
from .models import PromptRoute, RouteToken
from .service import RouteError, call, validate_inputs


class PromptEditorWidget(forms.Textarea):
    """The textarea, upgraded to struckdown's highlighting editor when the
    project serves the playground's static files under ``struckdown/``."""

    class Media:
        css = {"all": ("struckdown/css/codemirror-struckdown.css", "sd_routes/route-admin.css")}
        js = ("struckdown/js/codemirror-struckdown.js", "sd_routes/route-admin.js")

    def __init__(self, attrs=None):
        super().__init__({"rows": 18, "data-sd-editor": "1", **(attrs or {})})


class SchemaWidget(forms.Textarea):
    """JSON textarea with a button that fills it from the prompt as it stands."""

    def __init__(self, kind: str, attrs=None):
        super().__init__({"rows": 8, "class": "sd-schema", **(attrs or {})})
        self.kind = kind

    def render(self, name, value, attrs=None, renderer=None):
        return format_html(
            '{}<div class="sd-schema-tools"><button type="button" class="button sd-schema-generate"'
            ' data-kind="{}" data-url="{}">Generate from the prompt</button>'
            '<span class="sd-schema-status" role="status"></span></div>',
            super().render(name, value, attrs, renderer),
            self.kind,
            reverse("admin:sd_routes_promptroute_schema"),
        )


class PromptRouteForm(forms.ModelForm):
    class Meta:
        model = PromptRoute
        fields = "__all__"
        widgets = {
            "prompt": PromptEditorWidget(),
            "input_schema": SchemaWidget("input"),
            "output_schema": SchemaWidget("output"),
            "allowed_origins": forms.Textarea(attrs={"rows": 3}),
            "description": forms.Textarea(attrs={"rows": 2}),
        }


class RouteTokenForm(forms.ModelForm):
    rotate = forms.BooleanField(
        required=False, help_text="Replace this token; the old one stops working on save."
    )

    class Meta:
        model = RouteToken
        fields = ("name", "is_active")

    def save(self, commit=True):
        if self.cleaned_data.get("rotate"):
            self.instance.rotate()
        return super().save(commit)


class RouteTokenInline(admin.TabularInline):
    model = RouteToken
    form = RouteTokenForm
    extra = 0
    fields = ("name", "token", "is_active", "rotate", "last_used_at", "created_at")
    readonly_fields = ("token", "last_used_at", "created_at")


def _ensure_token(request, route) -> RouteToken:
    """The route's first active token, adding one called "default" if it has none.
    A route with no token cannot be called."""
    token = route.tokens.filter(is_active=True).first()
    if token is None:
        token = RouteToken.objects.create(route=route, name="default")
        messages.info(request, f'Added a token, "default": {token.token}')
    return token


class PromptRouteAdmin(admin.ModelAdmin):
    form = PromptRouteForm
    inlines = [RouteTokenInline]
    list_display = ("slug", "name", "model", "is_active", "window", "budget_today", "screen")
    list_filter = ("is_active", "model")
    search_fields = ("slug", "name", "description", "prompt")
    prepopulated_fields = {"slug": ("name",)}
    autocomplete_fields = ()
    readonly_fields = ("analysis", "budget_today", "endpoint")
    # the examples sit below the tokens (after_related_objects), not in a field
    change_form_template = "admin/sd_routes/promptroute/change_form.html"
    fieldsets = (
        (None, {"fields": ("name", "slug", "description", "is_active")}),
        ("Prompt", {"fields": ("prompt", "analysis")}),
        ("Model", {"fields": ("model", "credential")}),
        (
            "When and how much",
            {
                "fields": (
                    "opens_at",
                    "closes_at",
                    "daily_budget",
                    "budget_today",
                    "per_minute",
                    "max_input_tokens",
                )
            },
        ),
        (
            "Inputs and outputs",
            {
                "fields": (
                    "input_schema",
                    "output_schema",
                    "allow_images",
                    "max_image_bytes",
                    "max_images",
                )
            },
        ),
        ("Screening", {"fields": ("screen",)}),
        ("Calling it", {"fields": ("endpoint", "allowed_origins")}),
    )

    actions = ["rotate_tokens"]

    @admin.action(description="Rotate all tokens of the selected routes")
    def rotate_tokens(self, request, queryset):
        tokens = list(RouteToken.objects.filter(route__in=queryset))
        for token in tokens:
            token.rotate()
            token.save(update_fields=["token"])
        self.message_user(
            request, f"Replaced {len(tokens)} token(s); the old ones no longer work."
        )

    def save_related(self, request, form, formsets, change):
        """A route with no token cannot be called, so the first save gives it one."""
        super().save_related(request, form, formsets, change)
        _ensure_token(request, form.instance)

    # -- columns and read-only panels --------------------------------------

    @admin.display(description="Open")
    def window(self, obj):
        return "open" if obj.is_open() else "closed"

    @admin.display(description="Spent today")
    def budget_today(self, obj):
        if obj.pk is None:
            return "-"
        return f"${obj.spent_today():.2f} of ${obj.daily_budget:.2f}"

    @admin.display(description="Inputs and slots")
    def analysis(self, obj):
        if not obj.prompt:
            return "-"
        found = obj.analysis()
        if found["error"]:
            return format_html('<span class="errornote">{}</span>', found["error"])
        slots = format_html_join(", ", "<code>{}</code> ({})", ((s["name"], s["type"]) for s in found["slots"]))
        inputs = format_html_join(", ", "<code>{}</code>", ((i,) for i in found["inputs"]))
        return format_html("Inputs: {}<br>Slots: {}", inputs or "none", slots or "none")

    # the host is filled in by route-admin.js (sd-origin): a ModelAdmin is shared
    # across requests, so it has no request to build an absolute URL from
    @staticmethod
    def _endpoint(obj):
        return format_html(
            '<span class="sd-origin"></span>{}', reverse("sd_routes:call", args=[obj.slug])
        )

    @admin.display(description="Endpoint")
    def endpoint(self, obj):
        if obj.pk is None:
            return "Save to get the endpoint."
        playground = reverse("admin:sd_routes_promptroute_playground", args=[obj.pk])
        return format_html(
            '<code>POST {}</code> &nbsp; <a href="{}">Open the playground</a>',
            self._endpoint(obj),
            playground,
        )

    def change_view(self, request, object_id, form_url="", extra_context=None):
        route = self.get_object(request, object_id)
        if route is not None:
            # the examples need a working token; a route saved before tokens
            # were added by default may have none
            token = _ensure_token(request, route)
            key = token.token
            url = examples.ORIGIN + route.endpoint_path()
            extra_context = {
                **(extra_context or {}),
                "sd_examples": [
                    {
                        "title": "Web page",
                        "note": "A whole page: save it as an .html file and open it.",
                        "code": examples.html_page(route, url, key),
                    },
                    {
                        "title": "Python",
                        "note": "Needs requests (pip install requests).",
                        "code": examples.python_script(route, url, key),
                    },
                ],
            }
        return super().change_view(request, object_id, form_url, extra_context)

    # -- extra views --------------------------------------------------------

    def get_urls(self):
        return [
            path(
                "schema/",
                self.admin_site.admin_view(self.schema_view),
                name="sd_routes_promptroute_schema",
            ),
            path(
                "<int:pk>/playground/",
                self.admin_site.admin_view(self.playground_view),
                name="sd_routes_promptroute_playground",
            ),
            *super().get_urls(),
        ]

    def schema_view(self, request):
        """A starting schema from the prompt as typed (saved or not)."""
        if request.method != "POST":
            return JsonResponse({"error": "POST only."}, status=405)
        if not (self.has_add_permission(request) or self.has_change_permission(request)):
            raise PermissionDenied
        prompt = request.POST.get("prompt", "")
        found = schemas.analyse(prompt)
        if found["error"]:
            return JsonResponse({"error": f"The prompt does not parse: {found['error']}"}, status=400)
        if request.POST.get("kind") == "output":
            return JsonResponse({"schema": schemas.derived_output_schema(prompt)})
        try:
            tokens = int(request.POST.get("max_input_tokens") or 10_000)
        except ValueError:
            tokens = 10_000
        return JsonResponse({"schema": schemas.derived_input_schema(prompt, tokens)})

    def playground_view(self, request, pk):
        """Run the saved route on typed inputs, without a token. Counts
        against the route's budget like any other call."""
        route = get_object_or_404(PromptRoute, pk=pk)
        if not self.has_change_permission(request, route):
            raise PermissionDenied
        inputs = route.analysis()["inputs"]
        image_names = schemas.image_fields(route.effective_input_schema())
        values = {name: request.POST.get(name, "") for name in inputs if name not in image_names}
        outcome = None
        if request.method == "POST":
            outcome = self._run(request, route, values)
        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "route": route,
            "title": f"Playground: {route}",
            "fields": [
                {"name": n, "value": values.get(n, ""), "image": n in image_names} for n in inputs
            ],
            "outcome": outcome,
            "outcome_json": json.dumps(outcome, indent=2) if outcome else "",
            "change_url": reverse("admin:sd_routes_promptroute_change", args=[route.pk]),
        }
        return TemplateResponse(request, "admin/sd_routes/playground.html", context)

    def _run(self, request, route, values):
        if route.over_budget():
            messages.error(request, "Today's budget for this route is spent.")
            return None
        files = {k: request.FILES[k] for k in request.FILES}
        try:
            context = validate_inputs(route, values, files)
            return call(route, context, user=request.user, caller="admin playground")
        except RouteError as exc:
            return exc.as_json()


if not admin.site.is_registered(PromptRoute):
    admin.site.register(PromptRoute, PromptRouteAdmin)
