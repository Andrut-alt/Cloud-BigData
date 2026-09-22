"""A simple Debounce implementation.

Idea: postpone running a function until events stop arriving. Every new
event restarts the wait timer, so a burst of rapid calls collapses into
a single execution.

Options:
    leading  - run fn immediately on the first event of a burst.
    trailing - run fn once, after the burst ends, with the latest arguments.
"""

import threading


class Debouncer:
    """Creates debounced wrappers around functions.

    :param delay_ms: quiet period in milliseconds. fn runs only when no new
        call has arrived for this long.
    :param leading: run fn immediately on the first call of a burst.
    :param trailing: run fn after the burst ends, using the last call's
        arguments. With leading=True it only fires if there were further
        calls after the leading one.
    """

    def __init__(self, delay_ms, leading=False, trailing=True):
        if not leading and not trailing:
            raise ValueError("at least one of leading/trailing must be True")
        self.delay_ms = delay_ms
        self.leading = leading
        self.trailing = trailing

        self._lock = threading.Lock()
        self._timer = None
        self._pending = None  # (fn, args, kwargs) waiting for trailing run
        self._disposed = False

    def debounce(self, fn):
        """Return a wrapper around fn that is debounced by this Debouncer."""
        def wrapper(*args, **kwargs):
            self._on_call(fn, args, kwargs)

        wrapper.dispose = self.dispose
        return wrapper

    def dispose(self):
        """Cancel the pending timer and drop any pending trailing call.
        Later calls to the wrapper are ignored."""
        with self._lock:
            self._disposed = True
            self._pending = None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _on_call(self, fn, args, kwargs):
        run_now = False
        with self._lock:
            if self._disposed:
                return
            burst_active = self._timer is not None
            if not burst_active and self.leading:
                run_now = True
                self._pending = None
            else:
                self._pending = (fn, args, kwargs)

            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_ms / 1000, self._on_quiet)
            self._timer.daemon = True
            self._timer.start()

        if run_now:
            fn(*args, **kwargs)

    def _on_quiet(self):
        """Timer callback: the burst is over."""
        with self._lock:
            self._timer = None
            pending, self._pending = self._pending, None
            if self._disposed or not self.trailing or pending is None:
                return
        fn, args, kwargs = pending
        fn(*args, **kwargs)
