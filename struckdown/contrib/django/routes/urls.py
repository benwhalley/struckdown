"""Mount with ``path("api/", include("struckdown.contrib.django.routes.urls"))``."""

from django.urls import path

from .views import route_api

app_name = "sd_routes"

urlpatterns = [path("<slug:slug>/", route_api, name="call")]
