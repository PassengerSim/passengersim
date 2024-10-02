import contextlib

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
