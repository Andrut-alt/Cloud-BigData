"""A simple Throttle implementation (token bucket).

Idea: limit how often a function may run. A bucket holds up to `capacity`
tokens and refills at `refill_rate` tokens per second; every execution
spends one token. "N calls per interval" is the bucket with
capacity=N and refill_rate=N/interval.

Excess calls are either dropped or queued:
    drop  - calls without a token are discarded. With trailing=True the
            latest discarded call is kept and run once a token is free.
    queue - calls without a token wait and run later, in call order.

leading=True runs a call immediately when a token is available. With
leading=False nothing runs immediately: every call waits for the timer.
"""

import threading
import time
from collections import deque

DROP = "drop"
QUEUE = "queue"

_EPS = 1e-9


class Throttler:
    """Creates throttled wrappers around functions.

    Configure the limit with EITHER ``rate=(calls, interval_seconds)``
    OR ``capacity`` + ``refill_rate`` (tokens per second).

    :param leading: run a call immediately when a token is available.
    :param trailing: (drop mode) keep the latest rejected call and run it
        when a token becomes available.
    :param mode: "drop" or "queue".
    :param clock: time source in seconds (injectable for tests).
    """

    def __init__(self, rate=None, capacity=None, refill_rate=None,
                 leading=True, trailing=False, mode=DROP,
                 clock=time.monotonic):
        if rate is not None:
            if capacity is not None or refill_rate is not None:
                raise ValueError("use either rate or capacity/refill_rate")
            calls, interval = rate
            capacity, refill_rate = calls, calls / interval
        if not capacity or not refill_rate or capacity < 1 or refill_rate <= 0:
            raise ValueError("need rate or positive capacity and refill_rate")
        if mode not in (DROP, QUEUE):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == DROP and not leading and not trailing:
            raise ValueError("drop mode needs leading or trailing")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self.leading = leading
        self.trailing = trailing
        self.mode = mode
        self._clock = clock

        self._lock = threading.Lock()
        self._tokens = float(capacity)
        self._last_refill = clock()
        self._queue = deque()  # (fn, args, kwargs), used in queue mode
        self._pending = None   # latest rejected call, drop + trailing
        self._timer = None
        self._disposed = False

        self.executed = 0
        self.dropped = 0

    def throttle(self, fn):
        """Return a wrapper around fn limited by this Throttler."""
        def wrapper(*args, **kwargs):
            self._on_call(fn, args, kwargs)

        wrapper.dispose = self.dispose
        return wrapper

    def dispose(self):
        """Cancel the timer and discard waiting calls; later calls are
        ignored."""
        with self._lock:
            self._disposed = True
            self._queue.clear()
            self._pending = None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _refill(self):
        now = self._clock()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self.capacity,
                           self._tokens + elapsed * self.refill_rate)

    def _has_token(self):
        return self._tokens >= 1 - _EPS

    def _on_call(self, fn, args, kwargs):
        run_now = False
        with self._lock:
            if self._disposed:
                return
            self._refill()
            backlog = bool(self._queue)  # keep order: never jump the queue
            if self.leading and not backlog and self._has_token():
                self._tokens -= 1
                self.executed += 1
                run_now = True
            elif self.mode == QUEUE:
                self._queue.append((fn, args, kwargs))
            elif self.trailing:
                if self._pending is not None:
                    self.dropped += 1  # replaced by a newer call
                self._pending = (fn, args, kwargs)
            else:
                self.dropped += 1
            self._schedule_locked()

        if run_now:
            fn(*args, **kwargs)

    def _schedule_locked(self):
        """Start the timer if calls are waiting and none is running."""
        waiting = bool(self._queue) or self._pending is not None
        if not waiting or self._timer is not None or self._disposed:
            return
        missing = max(0.0, 1 - self._tokens)
        self._timer = threading.Timer(missing / self.refill_rate,
                                      self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self):
        """Run waiting calls while tokens last, then re-arm if needed."""
        ready = []
        with self._lock:
            self._timer = None
            if self._disposed:
                return
            self._refill()
            while self._has_token() and (self._queue or self._pending):
                self._tokens -= 1
                self.executed += 1
                if self._queue:
                    ready.append(self._queue.popleft())
                else:
                    ready.append(self._pending)
                    self._pending = None
            self._schedule_locked()

        for fn, args, kwargs in ready:
            fn(*args, **kwargs)
