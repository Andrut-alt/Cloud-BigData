"""A simple Timeout implementation.

Idea: cap how long an operation is allowed to run. If it does not finish
within timeout_ms, give up and raise OperationTimeoutError instead of
waiting forever.

Cancellation is cooperative: fn may accept a `cancel_event` keyword
argument (a threading.Event); it is set the moment the deadline expires,
so fn can check it periodically and stop early. Python cannot forcibly
kill a running thread, so a function that does not cooperate keeps
running in the background even after with_timeout() has raised -- its
eventual result is simply discarded by the caller.
"""

import inspect
import threading

_SENTINEL = object()


class OperationTimeoutError(Exception):
    """Raised when fn did not finish within timeout_ms."""


def with_timeout(fn, timeout_ms, *args, **kwargs):
    """Run fn(*args, **kwargs) and give up after timeout_ms milliseconds.

    If fn's signature accepts a `cancel_event` argument, a
    threading.Event is passed in and set as soon as the deadline expires.

    :raises OperationTimeoutError: fn did not finish in time.
    Any exception raised by fn within the deadline is re-raised as-is.
    """
    cancel_event = threading.Event()
    if _accepts_cancel_event(fn):
        kwargs = {**kwargs, "cancel_event": cancel_event}

    result = [_SENTINEL]
    error = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except BaseException as exc:  # re-raised on the caller's thread
            error[0] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_ms / 1000)

    if worker.is_alive():
        cancel_event.set()
        raise OperationTimeoutError(
            f"Operation exceeded timeout of {timeout_ms}ms")

    if error[0] is not None:
        raise error[0]
    return result[0]


def _accepts_cancel_event(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "cancel_event" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class Timeout:
    """A reusable, configured withTimeout -- handy for passing around (e.g.
    as the `fn` that Retry.execute() calls on each attempt).

    :param timeout_ms: deadline in milliseconds, applied on every run().
    """

    def __init__(self, timeout_ms):
        self.timeout_ms = timeout_ms

    def run(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) under this Timeout's deadline."""
        return with_timeout(fn, self.timeout_ms, *args, **kwargs)
