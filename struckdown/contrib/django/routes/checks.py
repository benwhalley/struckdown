"""System checks: the API's limits only hold with a shared cache."""

from django.conf import settings
from django.core.checks import Warning, register

_UNSHARED = {
    "django.core.cache.backends.dummy.DummyCache": "counts nothing, so no limit ever applies",
    "django.core.cache.backends.locmem.LocMemCache": "counts per process, so each worker gets its own limit",
}


@register()
def shared_cache(app_configs, **kwargs):
    backend = settings.CACHES.get("default", {}).get("BACKEND", "")
    if backend in _UNSHARED:
        return [
            Warning(
                f"API route rate limits use the default cache, and {backend} {_UNSHARED[backend]}.",
                hint="Use a shared cache such as Redis for the default cache.",
                id="sd_routes.W001",
            )
        ]
    return []
