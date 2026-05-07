"""Pytest defaults for struckdown tests.

The library is strict: callers must pass credentials explicitly. Tests opt in
to env-based credentials here once, so individual tests don't have to pass
``credentials=LLMCredentials.from_env()`` everywhere.
"""

from functools import wraps

import struckdown
from struckdown import LLMCredentials


def _with_default_creds(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.setdefault("credentials", LLMCredentials.from_env())
        return func(*args, **kwargs)

    return wrapper


def _async_with_default_creds(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        kwargs.setdefault("credentials", LLMCredentials.from_env())
        return await func(*args, **kwargs)

    return wrapper


struckdown.complete = _with_default_creds(struckdown.complete)
struckdown.complete_async = _async_with_default_creds(struckdown.complete_async)
struckdown.complete_incremental = _with_default_creds(struckdown.complete_incremental)
