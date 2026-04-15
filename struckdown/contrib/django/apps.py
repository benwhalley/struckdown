from django.apps import AppConfig


class SdModelsConfig(AppConfig):
    name = "struckdown.contrib.django"
    label = "sd_models"
    verbose_name = "Struckdown Models"
    default_auto_field = "django.db.models.BigAutoField"
