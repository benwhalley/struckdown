"""Default admin registration for sd_models.

Provides base admin classes for Credential, AvailableModel, and ModelSet.
Projects using a custom admin site should subclass these and register on
their own site.
"""

import logging

from django.contrib import admin, messages
from django.urls import path, reverse
from django.utils.html import format_html

from .models import AvailableModel, Credential, ModelSet

logger = logging.getLogger(__name__)


class CredentialAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "masked_key",
        "base_url",
        "pricing_source",
        "model_count",
        "is_active",
    ]
    list_filter = ["pricing_source", "is_active"]
    search_fields = ["name", "base_url"]
    readonly_fields = ["id", "created_at", "updated_at"]

    @admin.display(description="API Key")
    def masked_key(self, obj):
        return obj._mask_key()

    @admin.display(description="Models")
    def model_count(self, obj):
        return obj.models.count()


class AvailableModelAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "name",
        "model_name",
        "provider_display",
        "model_type",
        "credential",
        "data_residency",
        "input_cost_per_mtok",
        "output_cost_per_mtok",
        "is_active",
        "test_link",
    ]
    list_filter = ["model_type", "data_residency", "is_active", "credential"]
    list_editable = ["credential"]
    autocomplete_fields = ["credential"]
    search_fields = ["id", "name", "model_name"]
    readonly_fields = ["id", "created_at", "updated_at", "prices_updated_at"]
    actions = ["update_prices_action"]

    @admin.display(description="Provider")
    def provider_display(self, obj):
        return obj.provider_display

    @admin.display(description="Test")
    def test_link(self, obj):
        if obj.has_credentials:
            url = reverse(
                f"{self.admin_site.name}:sd_models_availablemodel_test",
                args=[obj.pk],
            )
            return format_html('<a href="{}">Test</a>', url)
        return "-"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "<path:object_id>/test/",
                self.admin_site.admin_view(self.test_connection_view),
                name="sd_models_availablemodel_test",
            ),
            path(
                "<path:object_id>/update-prices/",
                self.admin_site.admin_view(self.update_prices_view),
                name="sd_models_availablemodel_update_prices",
            ),
        ]
        return custom_urls + urls

    def change_view(self, request, object_id, form_url="", extra_context=None):
        extra_context = extra_context or {}
        site_name = self.admin_site.name
        extra_context["test_connection_url"] = reverse(
            f"{site_name}:sd_models_availablemodel_test", args=[object_id]
        )
        extra_context["update_prices_url"] = reverse(
            f"{site_name}:sd_models_availablemodel_update_prices", args=[object_id]
        )
        return super().change_view(request, object_id, form_url, extra_context)

    # -- update prices: single model --

    def update_prices_view(self, request, object_id):
        from django.http import HttpResponseRedirect

        model = self.get_object(request, object_id)
        if model is None:
            self.message_user(request, "Model not found.", messages.ERROR)
            return HttpResponseRedirect(request.META.get("HTTP_REFERER", "../"))

        if model.update_prices(force=True):
            self.message_user(
                request,
                f"Prices updated for {model.name}: "
                f"in=${model.input_cost_per_mtok}/Mtok, out=${model.output_cost_per_mtok}/Mtok",
                messages.SUCCESS,
            )
        else:
            self.message_user(
                request,
                f"Could not find pricing for {model.name} ({model.model_name}).",
                messages.WARNING,
            )

        return HttpResponseRedirect(
            reverse(
                f"{self.admin_site.name}:sd_models_availablemodel_change",
                args=[object_id],
            )
        )

    # -- update prices: queryset action --

    @admin.action(description="Update prices from pricing source")
    def update_prices_action(self, request, queryset):
        updated = 0
        failed = 0
        for model in queryset.select_related("credential"):
            if model.update_prices(force=True):
                updated += 1
            else:
                failed += 1
        self.message_user(
            request,
            f"Prices updated for {updated} model(s). {failed} not found.",
            messages.SUCCESS if failed == 0 else messages.WARNING,
        )

    # -- test connection --

    def test_connection_view(self, request, object_id):
        from django.http import HttpResponseRedirect

        model = self.get_object(request, object_id)
        if model is None:
            self.message_user(request, "Model not found.", messages.ERROR)
            return HttpResponseRedirect(request.META.get("HTTP_REFERER", "../"))

        if not model.has_credentials:
            self.message_user(
                request,
                f"No API key configured for {model.name}. Please add an API key first.",
                messages.ERROR,
            )
            return HttpResponseRedirect(
                reverse(
                    f"{self.admin_site.name}:sd_models_availablemodel_change",
                    args=[object_id],
                )
            )

        connection = None
        try:
            connection = self._build_test_connection(model)
            result = self._test_model(model)
            connection_debug = self._format_test_connection(connection)
            if result["success"]:
                if model.model_type == AvailableModel.ModelType.LLM:
                    self.message_user(
                        request,
                        "Connection successful! "
                        f"{connection_debug} Response: \"{result['response']}\" "
                        f"(cost: ${result['cost']:.6f})",
                        messages.SUCCESS,
                    )
                else:
                    self.message_user(
                        request,
                        "Connection successful! "
                        f"{connection_debug} Got {result['num_embeddings']} embeddings "
                        f"with {result['dimensions']} dimensions "
                        f"(cost: ${result['cost']:.6f})",
                        messages.SUCCESS,
                    )
            else:
                self.message_user(
                    request,
                    f"Connection failed: {result['error']} {connection_debug}",
                    messages.ERROR,
                )
        except Exception as e:
            logger.exception(f"Error testing model {model.model_name}")
            connection_debug = self._format_test_connection(connection)
            suffix = f" {connection_debug}" if connection_debug else ""
            self.message_user(
                request,
                f"Connection failed with exception: {e}{suffix}",
                messages.ERROR,
            )

        return HttpResponseRedirect(
            reverse(
                f"{self.admin_site.name}:sd_models_availablemodel_change",
                args=[object_id],
            )
        )

    @staticmethod
    def _obfuscate_api_key(api_key: str | None) -> str:
        if not api_key:
            return "(none)"
        if len(api_key) <= 10:
            return api_key[:2] + "***" + api_key[-2:]
        return api_key[:4] + "***" + api_key[-4:]

    def _build_test_connection(self, model: AvailableModel) -> dict:
        from struckdown.llm import LLM, LLMCredentials

        llm_obj = LLM(model_name=model.model_name)
        creds = LLMCredentials(api_key=model.api_key, base_url=model.base_url or None)
        pydantic_model = llm_obj.get_pydantic_model(creds)

        return {
            "creds": creds,
            "pydantic_model": pydantic_model,
            "endpoint": str(
                getattr(pydantic_model, "base_url", None)
                or creds.base_url
                or "(unknown)"
            ),
            "api_key_masked": self._obfuscate_api_key(
                creds.api_key_for_provider(model.provider) or creds.api_key
            ),
        }

    @staticmethod
    def _format_test_connection(connection: dict | None) -> str:
        if not connection:
            return ""
        return (
            f"[endpoint: {connection['endpoint']}, "
            f"api_key: {connection['api_key_masked']}]"
        )

    def _test_model(self, model: AvailableModel) -> dict:
        """Test the model with a simple query.

        Runs in a thread pool to avoid event loop conflicts with Django ASGI.
        """
        from concurrent.futures import ThreadPoolExecutor

        llm, credentials = model.get_llm_and_credentials()

        def _run_llm():
            from struckdown import complete

            result = complete(
                "Name a city that is sunny. Reply with just the city name. [[city]]",
                model=llm,
                credentials=credentials,
            )
            text = (
                result.outputs.get("city", "")
                if hasattr(result, "outputs")
                else str(result)
            )
            cost = getattr(result, "fresh_cost", 0) or 0
            return {"success": True, "response": str(text)[:100], "cost": cost}

        def _run_embedding():
            from struckdown import get_embedding

            embeddings = get_embedding(
                ["Hello world"],
                model=model.bare_model_name,
                credentials=credentials,
            )
            return {
                "success": True,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "num_embeddings": len(embeddings),
                "cost": getattr(embeddings, "total_cost", 0) or 0,
            }

        fn = (
            _run_llm
            if model.model_type == AvailableModel.ModelType.LLM
            else _run_embedding
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(fn).result(timeout=30)


class ModelSetAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "is_default",
        "default_credential",
        "model_count",
        "data_residency_display",
        "is_active",
        "created_at",
    ]
    list_filter = ["is_default", "is_active"]
    search_fields = ["name"]
    filter_horizontal = ["available_models"]
    readonly_fields = ["id", "created_at", "updated_at"]

    @admin.display(description="Models")
    def model_count(self, obj):
        return obj.available_models.count()

    @admin.display(description="Data Residency")
    def data_residency_display(self, obj):
        return obj.data_residency_summary


# Only register on default admin site. Projects with custom admin sites
# (like soakresearch's OTPAdminSite) should register manually.
if not admin.site.is_registered(Credential):
    admin.site.register(Credential, CredentialAdmin)
if not admin.site.is_registered(AvailableModel):
    admin.site.register(AvailableModel, AvailableModelAdmin)
if not admin.site.is_registered(ModelSet):
    admin.site.register(ModelSet, ModelSetAdmin)
