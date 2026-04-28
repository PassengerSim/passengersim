import contextlib
from functools import wraps

try:
    import wakepy
except ImportError:
    wakepy = None


@contextlib.contextmanager
def keep_awake():
    """
    Keep the computer awake while in this context.
    """
    if wakepy is None:
        yield
        return

    with wakepy.keep.running():
        yield


# decorator version of keep_awake
def keep_alive(func):
    @wraps(func)  # Ensure original function keeps its metadata
    def wrapper(*args, **kwargs):
        with keep_awake():
            return func(*args, **kwargs)

    return wrapper
