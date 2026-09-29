from django.apps import AppConfig


class SdModelsConfig(AppConfig):
    name = "struckdown.contrib.django"
    label = "sd_models"
    verbose_name = "Struckdown Models"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        # every call struckdown makes in this process lands in the ledger
        from struckdown.ledger import register_usage_handler

        from .ledger import handler

        register_usage_handler(handler)
