"""Default admin registration for sd_models.

Projects using a custom admin site (e.g. OTP admin) should register
AvailableModel and ModelSet on their own site and skip this module.
"""

from django.contrib import admin

from .models import AvailableModel, ModelSet


class AvailableModelAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "name",
        "model_name",
        "provider_display",
        "model_type",
        "data_residency",
        "is_active",
    ]
    list_filter = ["model_type", "data_residency", "is_active"]
    search_fields = ["id", "name", "model_name"]
    readonly_fields = ["id", "created_at", "updated_at"]

    @admin.display(description="Provider")
    def provider_display(self, obj):
        return obj.provider_display


class ModelSetAdmin(admin.ModelAdmin):
    list_display = ["name", "is_default", "model_count", "data_residency_summary", "is_active"]
    list_filter = ["is_default", "is_active"]
    search_fields = ["name"]
    filter_horizontal = ["available_models"]
    readonly_fields = ["id", "created_at", "updated_at"]

    @admin.display(description="Models")
    def model_count(self, obj):
        return obj.available_models.count()


# Only register on default admin site. Projects with custom admin sites
# (like soakresearch's OTPAdminSite) should register manually.
if not admin.site.is_registered(AvailableModel):
    admin.site.register(AvailableModel, AvailableModelAdmin)
if not admin.site.is_registered(ModelSet):
    admin.site.register(ModelSet, ModelSetAdmin)
