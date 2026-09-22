"""A simple Circuit Breaker implementation.

Idea: stop calling a remote/unreliable operation once it fails too many
times in a row, give it time to recover, then cautiously try again.

State machine:
    CLOSED --(N consecutive failures)--> OPEN
    OPEN --(open_state_duration elapsed)--> HALF_OPEN
    HALF_OPEN --(all trial calls succeed)--> CLOSED
    HALF_OPEN --(any trial call fails)--> OPEN
"""

import time
from enum import Enum


class State(Enum):
    """The three possible states of the circuit."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is open, or the
    half-open trial-call quota is already used up."""


class CallTimeoutError(Exception):
    """Raised when a wrapped call takes longer than timeout_per_call."""


class CircuitBreaker:
    """Wraps calls to a potentially failing operation and stops calling it
    once it fails too often, protecting the caller from cascading failures.

    :param failure_threshold: consecutive failures (in CLOSED state)
        needed to trip the circuit to OPEN.
    :param half_open_max_calls: number of trial calls allowed in
        HALF_OPEN state. Any failure among them re-opens the circuit;
        if all of them succeed the circuit closes again.
    :param open_state_duration: seconds to stay OPEN before allowing a
        trial move to HALF_OPEN.
    :param timeout_per_call: max seconds a single call to `fn` may take.
        Going over this counts as a failure.
    """

    def __init__(self, failure_threshold=3, half_open_max_calls=1,
                 open_state_duration=10.0, timeout_per_call=5.0):
        self.failure_threshold = failure_threshold
        self.half_open_max_calls = half_open_max_calls
        self.open_state_duration = open_state_duration
        self.timeout_per_call = timeout_per_call

        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

        # Only used while in HALF_OPEN.
        self._half_open_calls_made = 0
        self._half_open_successes = 0

    def state(self):
        """Return the current state as a string (e.g. "CLOSED")."""
        self._expire_open_state_if_due()
        return self._state.value

    def call(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) through the circuit breaker.

        :raises CircuitBreakerOpenError: the circuit is OPEN, or the
            HALF_OPEN trial-call quota is already used up.
        :raises CallTimeoutError: fn took longer than timeout_per_call.
        Any other exception raised by fn is re-raised as-is.
        """
        self._expire_open_state_if_due()

        if self._state == State.OPEN:
            raise CircuitBreakerOpenError("Circuit is open")

        if self._state == State.HALF_OPEN:
            if self._half_open_calls_made >= self.half_open_max_calls:
                raise CircuitBreakerOpenError("Half-open quota reached")
            self._half_open_calls_made += 1

        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise

        # fn returned normally, but did it take too long?
        if time.monotonic() - start > self.timeout_per_call:
            self._on_failure()
            raise CallTimeoutError(
                f"Call exceeded timeout of {self.timeout_per_call}s")

        self._on_success()
        return result

    def _on_success(self):
        if self._state == State.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_max_calls:
                self._close()
        else:
            self._consecutive_failures = 0

    def _on_failure(self):
        if self._state == State.HALF_OPEN:
            self._open()
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._open()

    def _open(self):
        self._state = State.OPEN
        self._opened_at = time.monotonic()
        self._consecutive_failures = 0
        self._reset_half_open_counters()

    def _close(self):
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._reset_half_open_counters()

    def _reset_half_open_counters(self):
        self._half_open_calls_made = 0
        self._half_open_successes = 0

    def _expire_open_state_if_due(self):
        """Move OPEN -> HALF_OPEN once open_state_duration has passed."""
        if self._state == State.OPEN:
            if time.monotonic() - self._opened_at >= self.open_state_duration:
                self._state = State.HALF_OPEN
                self._reset_half_open_counters()