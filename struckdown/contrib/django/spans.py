"""Spans: what a call belongs to.

A span is held in a context variable as a :class:`SpanHandle`, which is cheap
and needs no database: the ``LLMSpan`` row is only written when the first
call inside it is recorded (or on open, for a span a caller named). A root
span is opened for every HTTP request by :class:`SpanMiddleware` and for
every Celery task by :func:`install_celery_hooks`, so every call is
attributed to something without any call site changing; a request that makes
no LLM call costs nothing.

The handle is mutated in place rather than replaced. Context variables are
copied into ``sync_to_async`` threads and asyncio tasks, so a row created
there would be invisible to the caller if it were stored by reassigning the
variable; a shared object is seen by everyone holding it.

Explicit spans::

    with llm_span("hub_assistant.turn", user=request.user, obj=session):
        ...

    span = open_span("tower.assist.turn", user=user)   # for an async generator
    ...
    close_span(span)

From async code use ``aopen_span`` / ``aclose_span`` (the same, with the
database work on a thread).
"""

from __future__ import annotations

import logging
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class SpanHandle:
    name: str
    parent: Optional["SpanHandle"] = None
    user: Any = None  # a user instance, a pk, or a callable returning either
    obj: Any = None
    attributes: dict = field(default_factory=dict)
    capture_payloads: bool = False
    started_at: datetime = field(default_factory=timezone.now)
    ended_at: Optional[datetime] = None
    # a callable that gives the span its final name late, for a root span whose
    # request has not been routed yet when the middleware opens it
    name_fn: Any = None
    row: Any = None  # the LLMSpan once written

    @property
    def root_name(self) -> str:
        handle = self
        while handle.parent is not None:
            handle = handle.parent
        return handle.resolved_name

    @property
    def resolved_name(self) -> str:
        if self.name_fn is not None:
            try:
                name = self.name_fn()
            except Exception:
                name = None
            if name:
                self.name = name
                self.name_fn = None
        return self.name


_current: ContextVar[Optional[SpanHandle]] = ContextVar("struckdown_span", default=None)


def current_span() -> Optional[SpanHandle]:
    return _current.get()


def _user_pk(user) -> Optional[Any]:
    if callable(user):
        user = user()
    if user is None:
        return None
    if hasattr(user, "is_authenticated"):
        return user.pk if user.is_authenticated else None
    return user


def materialise(handle: SpanHandle):
    """The ``LLMSpan`` row for ``handle``, written now if it was not before."""
    from django.contrib.contenttypes.models import ContentType

    from .models import LLMSpan

    if handle.row is not None:
        return handle.row
    parent_row = materialise(handle.parent) if handle.parent is not None else None
    obj = handle.obj
    content_type = ContentType.objects.get_for_model(obj) if obj is not None else None
    handle.row = LLMSpan.objects.create(
        name=handle.resolved_name[:160],
        parent=parent_row,
        user_id=_user_pk(handle.user),
        content_type=content_type,
        object_id=str(obj.pk) if obj is not None else "",
        started_at=handle.started_at,
        ended_at=handle.ended_at,
        attributes=handle.attributes,
        capture_payloads=handle.capture_payloads,
    )
    return handle.row


def _close_row(handle: SpanHandle):
    if handle.row is not None and handle.row.ended_at is None:
        from .models import LLMSpan

        LLMSpan.objects.filter(pk=handle.row.pk).update(ended_at=handle.ended_at)
        handle.row.ended_at = handle.ended_at


def open_span(
    name: str,
    *,
    user=None,
    obj=None,
    capture: bool = False,
    eager: bool = True,
    root: bool = False,
    name_fn=None,
    **attributes,
) -> SpanHandle:
    """Open a span in this context and return its handle.

    ``eager`` writes the row now, so a span a caller named survives a crash
    inside it; root spans pass ``eager=False`` and are written on first use.
    ``root`` ignores any handle already in the context: a request or task is
    never a child of whatever a reused thread was last doing.
    """
    handle = SpanHandle(
        name=name,
        parent=None if root else _current.get(),
        user=user,
        obj=obj,
        attributes=attributes,
        capture_payloads=capture,
        name_fn=name_fn,
    )
    handle._token = _current.set(handle)
    if eager:
        try:
            materialise(handle)
        except Exception:
            logger.exception("could not open span %s", name)
    return handle


def close_span(handle: SpanHandle) -> None:
    handle.ended_at = timezone.now()
    try:
        _close_row(handle)
    except Exception:
        logger.exception("could not close span %s", handle.name)
    token = getattr(handle, "_token", None)
    if token is not None:
        try:
            _current.reset(token)
        except ValueError:
            # closed in a different context from the one that opened it (an
            # async generator finalised elsewhere); the parent is still right
            _current.set(handle.parent)
        handle._token = None


@contextmanager
def llm_span(name: str, *, user=None, obj=None, capture: bool = False, **attributes):
    handle = open_span(name, user=user, obj=obj, capture=capture, **attributes)
    try:
        yield handle
    finally:
        close_span(handle)


async def aopen_span(name: str, *, user=None, obj=None, capture: bool = False, **attributes):
    """``open_span`` from async code: the row is written on a thread."""
    handle = SpanHandle(
        name=name,
        parent=_current.get(),
        user=user,
        obj=obj,
        attributes=attributes,
        capture_payloads=capture,
    )
    handle._token = _current.set(handle)
    try:
        await sync_to_async(materialise, thread_sensitive=True)(handle)
    except Exception:
        logger.exception("could not open span %s", name)
    return handle


async def aclose_span(handle: SpanHandle) -> None:
    handle.ended_at = timezone.now()
    try:
        await sync_to_async(_close_row, thread_sensitive=True)(handle)
    except Exception:
        logger.exception("could not close span %s", handle.name)
    token = getattr(handle, "_token", None)
    if token is not None:
        try:
            _current.reset(token)
        except ValueError:
            _current.set(handle.parent)
        handle._token = None


# -- root spans: one per HTTP request ------------------------------------------


def _request_span_name(request_ref):
    def name():
        request = request_ref()
        if request is None:
            return None
        match = getattr(request, "resolver_match", None)
        if match is not None and match.view_name:
            return f"http:{match.view_name}"
        return f"http:{request.path}"

    return name


def _request_user(request_ref):
    def user():
        request = request_ref()
        return getattr(request, "user", None) if request is not None else None

    return user


class SpanMiddleware:
    """Opens a lazy root span per request, named after the resolved view.

    Sync and async capable, so it never forces an async view chain onto a
    thread. Sits after ``AuthenticationMiddleware`` so the user is known. No
    database work happens here: the row is written only if a call is made.

    When the view returns, the span context is reset to what it was before
    the request, which drops any span the view opened and left open. A view
    that streams should open its span inside the body.
    """

    sync_capable = True
    async_capable = True

    def __init__(self, get_response):
        self.get_response = get_response
        if iscoroutinefunction(get_response):
            markcoroutinefunction(self)

    def _open(self, request) -> SpanHandle:
        ref = weakref.ref(request)
        return open_span(
            f"http:{request.path}",
            user=_request_user(ref),
            eager=False,
            root=True,
            name_fn=_request_span_name(ref),
            method=request.method,
            path=request.path[:200],
        )

    @staticmethod
    def _finish(handle: SpanHandle, response) -> None:
        """The view has returned. A streaming body is still to come, so its
        iterator is wrapped to run under the handle; a plain response ends the
        span here."""
        handle.ended_at = timezone.now()
        token = getattr(handle, "_token", None)
        if token is not None:
            try:
                _current.reset(token)
            except ValueError:
                _current.set(None)
            handle._token = None
        if response is not None and getattr(response, "streaming", False):
            _wrap_streaming(response, handle)

    def __call__(self, request):
        if iscoroutinefunction(self):
            return self.__acall__(request)
        handle = self._open(request)
        response = None
        try:
            response = self.get_response(request)
        finally:
            self._finish(handle, response)
            try:
                _close_row(handle)
            except Exception:
                logger.exception("could not close request span")
        return response

    async def __acall__(self, request):
        handle = self._open(request)
        response = None
        try:
            response = await self.get_response(request)
        finally:
            self._finish(handle, response)
            if handle.row is not None:
                try:
                    await sync_to_async(_close_row, thread_sensitive=True)(handle)
                except Exception:
                    logger.exception("could not close request span")
        return response


def _wrap_streaming(response, handle: SpanHandle) -> None:
    """Run a streaming body under ``handle``, chunk by chunk.

    The body is produced after the middleware has returned, in whatever
    context iterates it, so the handle is set around each step of the body
    (``next()`` or ``__anext__()``, while it computes a chunk) and reset
    before the chunk is passed on. It is never left in the consumer's
    context, where a reused thread would carry it into the next request.

    A span the body opens itself is carried from one step to the next, so it
    stays current across chunks until the body closes it.
    """
    from django.http import FileResponse

    if isinstance(response, FileResponse):
        return

    if response.is_async:
        source = response.streaming_content

        async def agen():
            iterator = source.__aiter__()
            inner = handle
            try:
                while True:
                    token = _current.set(inner)
                    try:
                        chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        return
                    finally:
                        inner = _current.get()
                        _current.reset(token)
                    yield chunk
            finally:
                # a consumer that stops early closes the body under the span too
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    token = _current.set(inner)
                    try:
                        await aclose()
                    finally:
                        _current.reset(token)

        response.streaming_content = agen()
        return

    source = response.streaming_content

    def gen():
        iterator = iter(source)
        inner = handle
        while True:
            token = _current.set(inner)
            try:
                chunk = next(iterator)
            except StopIteration:
                return
            finally:
                inner = _current.get()
                _current.reset(token)
            yield chunk

    response.streaming_content = gen()


# -- root spans: one per Celery task --------------------------------------------

_task_handles: dict = {}


def _task_prerun(sender=None, task_id=None, task=None, **kwargs):
    name = getattr(task, "name", None) or getattr(sender, "name", "task")
    _task_handles[task_id] = open_span(f"celery:{name}", eager=False, root=True, task_id=task_id)


def _task_postrun(sender=None, task_id=None, **kwargs):
    handle = _task_handles.pop(task_id, None)
    if handle is not None:
        close_span(handle)


def install_celery_hooks() -> None:
    """Open a root span for every task. Call once, where the Celery app is built."""
    from celery.signals import task_postrun, task_prerun

    task_prerun.connect(_task_prerun, weak=False, dispatch_uid="struckdown_ledger_prerun")
    task_postrun.connect(_task_postrun, weak=False, dispatch_uid="struckdown_ledger_postrun")
