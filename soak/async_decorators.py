import contextvars
import functools
import logging
from typing import Any, Callable, Optional

import anyio

logger = logging.getLogger(__name__)
_current_task_group: contextvars.ContextVar[Optional[anyio.abc.TaskGroup]] = contextvars.ContextVar(
    "task_group", default=None
)


# exception classes for backward compatibility with wellspring models
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


def get_run_context() -> Optional[anyio.abc.TaskGroup]:
    """get current task group for compatibility with existing code"""
    return _current_task_group.get()


def flow(func: Callable) -> Callable:
    """anyio-based @flow decorator with cancellation support."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        async with anyio.create_task_group() as tg:
            token = _current_task_group.set(tg)
            try:
                return await func(*args, **kwargs)
            except anyio.get_cancelled_exc_class():
                logger.warning(f"Flow {func.__name__} cancelled")
                raise CancelledRun(f"Flow {func.__name__} was cancelled")
            finally:
                _current_task_group.reset(token)

    def sync_wrapper(*args, **kwargs):
        return anyio.run(async_wrapper, *args, **kwargs)

    async_wrapper.sync = sync_wrapper
    return async_wrapper


def task(func: Callable) -> Callable:
    """anyio-based @task decorator converts a function to coroutine."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # tasks run within their current cancellation context
        # structured concurrency is provided by the @flow decorator's task group
        try:
            return await func(*args, **kwargs)
        except anyio.get_cancelled_exc_class():
            logger.warning(f"Task {func.__name__} cancelled")
            raise Cancelled(f"Task {func.__name__} was cancelled")

    return async_wrapper


def cancel_flow():
    """cancel current flow by cancelling the task group"""
    tg = _current_task_group.get()
    if tg:
        tg.cancel_scope.cancel()
        logger.info("Cancelled current flow task group")
