from django.contrib import admin
from django.http import HttpResponse, StreamingHttpResponse
from django.urls import path

from struckdown.contrib.django import ledger
from struckdown.contrib.django.spans import current_span
from struckdown.ledger import UsageRecord


def _maybe_call(request):
    """``?call=1`` makes the view record one usage record, as an LLM call would."""
    if request.GET.get("call"):
        ledger.write_record(UsageRecord(kind="chat", model_name="stub", input_tokens=10))


def sync_view(request):
    handle = current_span()
    _maybe_call(request)
    return HttpResponse(handle.resolved_name if handle else "none")


async def async_view(request):
    handle = current_span()
    return HttpResponse(handle.resolved_name if handle else "none")


def streaming_view(request):
    def body():
        handle = current_span()
        yield (handle.resolved_name if handle else "none").encode()

    return StreamingHttpResponse(body())


urlpatterns = [
    path("admin/", admin.site.urls),
    path("sync/", sync_view, name="sync_view"),
    path("async/", async_view, name="async_view"),
    path("stream/", streaming_view, name="streaming_view"),
]
