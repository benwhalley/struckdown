from django.apps import AppConfig


class SdRoutesConfig(AppConfig):
    """Struckdown prompts served as small JSON APIs (``/api/<slug>/``).

    Optional: add ``struckdown.contrib.django.routes`` to INSTALLED_APPS after
    ``struckdown.contrib.django``, and include ``struckdown.contrib.django.routes.urls``.
    """

    name = "struckdown.contrib.django.routes"
    label = "sd_routes"
    verbose_name = "Struckdown API routes"
    default_auto_field = "django.db.models.BigAutoField"
