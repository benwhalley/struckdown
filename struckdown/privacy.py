"""Private calls: nothing they send or receive is kept by struckdown.

Inside :func:`private`, calls neither read nor write the response cache, and
no usage handler is given their request or response bodies, whatever any
handler or setting asks for. The usage record itself (tokens, cost, timing)
is still emitted, so spend stays visible.

The flag is a context variable, so it reaches everything the block starts:
nested blocks, ``sync_to_async`` threads and asyncio tasks copied from it. A
nested block cannot switch it off.

For input that must not be kept anywhere outside the host's own storage
(student records, health data)::

    with struckdown.private():
        struckdown.complete(...)
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

_private: ContextVar[bool] = ContextVar("struckdown_private", default=False)


def is_private() -> bool:
    return _private.get()


def enter_private() -> Token:
    """Make this context private until :func:`exit_private` is given the token."""
    return _private.set(True)


def exit_private(token: Token) -> None:
    _private.reset(token)


@contextmanager
def private() -> Iterator[None]:
    token = enter_private()
    try:
        yield
    finally:
        exit_private(token)
