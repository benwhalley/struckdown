"""Spans: nesting, users, objects, and the root spans the middleware and Celery open."""

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model

from struckdown.contrib.django import ledger, spans
from struckdown.contrib.django.models import LLMCall, LLMSpan
from struckdown.contrib.django.spans import (aclose_span, aopen_span,
                                             close_span, current_span,
                                             llm_span, open_span)
from struckdown.ledger import UsageRecord

pytestmark = pytest.mark.django_db


def _emit():
    return ledger.write_record(UsageRecord(kind="chat", model_name="stub", input_tokens=10))


def test_nested_spans_record_their_parent_and_the_root_name():
    with llm_span("outer") as outer:
        with llm_span("inner") as inner:
            call = _emit()
    assert inner.row.parent == outer.row
    assert call.span == inner.row
    assert call.root_name == "outer"
    assert call.span.root_name == "outer"


def test_a_named_span_is_written_on_open_and_closed_on_exit():
    with llm_span("eager") as handle:
        assert handle.row is not None
        assert handle.row.ended_at is None
    handle.row.refresh_from_db()
    assert handle.row.ended_at is not None


def test_span_carries_user_and_object():
    user = get_user_model().objects.create(username="u")
    with llm_span("with-obj", user=user, obj=user, colour="blue") as handle:
        pass
    row = LLMSpan.objects.get(pk=handle.row.pk)
    assert row.user == user
    assert row.obj == user
    assert row.attributes == {"colour": "blue"}


def test_an_anonymous_user_is_recorded_as_nobody():
    from django.contrib.auth.models import AnonymousUser

    with llm_span("anon", user=AnonymousUser()) as handle:
        pass
    assert handle.row.user_id is None


def test_open_and_close_without_a_with_block():
    handle = open_span("gen")
    assert current_span() is handle
    close_span(handle)
    assert current_span() is None


@pytest.mark.django_db(transaction=True)
def test_async_open_and_close():
    async def go():
        handle = await aopen_span("async-span")
        assert current_span() is handle
        assert handle.row is not None
        await aclose_span(handle)
        assert current_span() is None
        return handle

    handle = async_to_sync(go)()
    assert LLMSpan.objects.get(pk=handle.row.pk).ended_at is not None


class TestRequestSpans:
    def test_a_request_without_calls_writes_no_span(self, client):
        response = client.get("/sync/")
        assert response.status_code == 200
        assert response.content == b"http:sync_view"
        assert LLMSpan.objects.count() == 0

    def test_a_call_inside_a_request_is_attributed_to_the_view(self, client):
        client.get("/sync/?call=1")
        call = LLMCall.objects.get()
        assert call.root_name == "http:sync_view"
        assert call.span.name == "http:sync_view"
        assert call.span.attributes["method"] == "GET"
        assert call.span.ended_at is not None

    def test_the_request_user_lands_on_the_span(self, client, django_user_model):
        user = django_user_model.objects.create_user(username="ben", password="pw")
        client.force_login(user)
        client.get("/sync/?call=1")
        assert LLMSpan.objects.get().user == user

    def test_a_request_span_is_a_root_even_after_a_stale_handle(self, client):
        # a WSGI thread reused after a streaming response still holds the old
        # handle; the next request must not nest under it
        open_span("stale", eager=False)
        client.get("/sync/?call=1")
        assert LLMSpan.objects.get().parent is None

    @pytest.mark.django_db(transaction=True)
    def test_the_async_path_names_the_view_too(self, async_client):
        response = async_to_sync(async_client.get)("/async/")
        assert response.status_code == 200
        assert response.content == b"http:async_view"

    def test_a_streaming_body_still_sees_the_span(self, client):
        response = client.get("/stream/")
        assert b"".join(response.streaming_content) == b"http:streaming_view"

    @pytest.mark.django_db(transaction=True)
    def test_calls_an_async_streaming_body_makes_carry_the_request_span(self, async_client):
        async def consume():
            response = await async_client.get("/astream/")
            chunks = [chunk async for chunk in response.streaming_content]
            # the consumer's own context never holds the request's handle
            assert current_span() is None
            return b"".join(chunks)

        body = async_to_sync(consume)()
        assert body == b"http:async_streaming_view;http:async_streaming_view;"
        calls = LLMCall.objects.order_by("model_name")
        assert [c.model_name for c in calls] == ["stub-0", "stub-1"]
        assert {c.span.name for c in calls} == {"http:async_streaming_view"}

    @pytest.mark.django_db(transaction=True)
    def test_a_span_an_async_body_opens_lasts_across_its_chunks(self, async_client):
        async def consume():
            response = await async_client.get("/astream/?own=1")
            return b"".join([chunk async for chunk in response.streaming_content])

        assert async_to_sync(consume)() == b"body;body;"
        calls = LLMCall.objects.all()
        assert {c.span.name for c in calls} == {"body"}
        assert {c.root_name for c in calls} == {"http:async_streaming_view"}


class TestCelerySpans:
    def test_prerun_and_postrun_open_and_close_a_root_span(self):
        class Task:
            name = "app.tasks.do_thing"

        spans._task_prerun(sender=Task(), task_id="t1", task=Task())
        assert current_span().name == "celery:app.tasks.do_thing"
        call = _emit()
        spans._task_postrun(sender=Task(), task_id="t1")
        assert current_span() is None
        assert call.root_name == "celery:app.tasks.do_thing"
        assert call.span.attributes["task_id"] == "t1"
        assert LLMSpan.objects.get().ended_at is not None

    def test_a_task_without_calls_writes_no_span(self):
        class Task:
            name = "app.tasks.quiet"

        spans._task_prerun(sender=Task(), task_id="t2", task=Task())
        spans._task_postrun(sender=Task(), task_id="t2")
        assert LLMSpan.objects.count() == 0
