from django.contrib import admin
from django.http import HttpResponse, StreamingHttpResponse
from django.urls import path

from struckdown.contrib.django import ledger
from struckdown.contrib.django.spans import aclose_span, aopen_span, current_span
from struckdown.ledger import UsageRecord, emit_async


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


def async_streaming_view(request):
    """An async body that makes a call before each chunk, as a streamed
    completion does; ``?own=1`` opens its own span first and keeps it open
    across chunks."""

    async def body():
        own = await aopen_span("body") if request.GET.get("own") else None
        for n in range(2):
            await emit_async(UsageRecord(kind="chat", model_name=f"stub-{n}"))
            handle = current_span()
            yield (handle.name if handle else "none").encode() + b";"
        if own is not None:
            await aclose_span(own)

    return StreamingHttpResponse(body())


urlpatterns = [
    path("admin/", admin.site.urls),
    path("sync/", sync_view, name="sync_view"),
    path("async/", async_view, name="async_view"),
    path("stream/", streaming_view, name="streaming_view"),
    path("astream/", async_streaming_view, name="async_streaming_view"),
]
