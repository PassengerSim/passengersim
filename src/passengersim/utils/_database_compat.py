"""Compatibility adapters for the former database connection shorthand."""

import warnings
from functools import wraps
from inspect import signature


def deprecated_cnx_keyword(name="db", *, stacklevel=2):
    """Accept the old keyword without changing positional argument behavior."""

    def decorate(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if "cnx" in kwargs:
                if name in kwargs:
                    raise TypeError(f"{func.__qualname__} received both {name}= and cnx=")
                kwargs[name] = kwargs.pop("cnx")
                # Bind before warning to catch a duplicate positional argument too.
                sig.bind_partial(*args, **kwargs)
                warnings.warn(
                    f"cnx= is deprecated; use {name}= instead.",
                    DeprecationWarning,
                    stacklevel=stacklevel,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorate
