"""Tests for the CircuitBreaker class.

Time-dependent scenarios (open_state_duration, timeout_per_call) use small
real values with short time.sleep calls, keeping the suite fast while
exercising real timing behaviour.
"""

import time

import pytest

from circuit_breaker import (
    CallTimeoutError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    State,
)


def ok(*_args, **_kwargs):
    """A dependency call that always succeeds."""
    return "ok"


def fail(*_args, **_kwargs):
    """A dependency call that always raises."""
    raise RuntimeError("boom")


def hang(seconds):
    """Build a dependency call that sleeps longer than any timeout used
    in the tests, simulating an unresponsive remote service."""
    def _hang(*_args, **_kwargs):
        time.sleep(seconds)
        return "too late"
    return _hang


class TestClosedState:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state() == State.CLOSED.value

    def test_successful_calls_keep_circuit_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(5):
            assert cb.call(ok) == "ok"
        assert cb.state() == State.CLOSED.value

    def test_opens_after_consecutive_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, open_state_duration=60)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(fail)
        assert cb.state() == State.OPEN.value

    def test_success_resets_consecutive_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, open_state_duration=60)
        with pytest.raises(RuntimeError):
            cb.call(fail)
        with pytest.raises(RuntimeError):
            cb.call(fail)
        cb.call(ok)  # resets the streak
        with pytest.raises(RuntimeError):
            cb.call(fail)
        # Only 1 consecutive failure since the reset -> still closed.
        assert cb.state() == State.CLOSED.value

    def test_timeout_counts_as_failure_and_can_trip_circuit(self):
        cb = CircuitBreaker(failure_threshold=2, timeout_per_call=0.1,
                             open_state_duration=60)
        slow = hang(0.5)
        with pytest.raises(CallTimeoutError):
            cb.call(slow)
        with pytest.raises(CallTimeoutError):
            cb.call(slow)
        assert cb.state() == State.OPEN.value


class TestOpenState:
    def test_calls_are_blocked_while_open(self):
        cb = CircuitBreaker(failure_threshold=1, open_state_duration=60)
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state() == State.OPEN.value

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(ok)

    def test_transitions_to_half_open_after_duration_elapses(self):
        cb = CircuitBreaker(failure_threshold=1, open_state_duration=0.1)
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state() == State.OPEN.value

        time.sleep(0.15)
        assert cb.state() == State.HALF_OPEN.value


class TestHalfOpenState:
    def _open_then_wait(self, **kwargs):
        cb = CircuitBreaker(failure_threshold=1, open_state_duration=0.1,
                             **kwargs)
        with pytest.raises(RuntimeError):
            cb.call(fail)
        time.sleep(0.15)
        assert cb.state() == State.HALF_OPEN.value
        return cb

    def test_closes_after_required_successful_trial_calls(self):
        cb = self._open_then_wait(half_open_max_calls=2)
        assert cb.call(ok) == "ok"
        assert cb.state() == State.HALF_OPEN.value  # 1 of 2 done
        assert cb.call(ok) == "ok"
        assert cb.state() == State.CLOSED.value

    def test_reopens_on_any_failed_trial_call(self):
        cb = self._open_then_wait(half_open_max_calls=3)
        cb.call(ok)  # one success so far
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state() == State.OPEN.value

    def test_limits_number_of_trial_calls(self):
        cb = self._open_then_wait(half_open_max_calls=1)
        assert cb.call(ok) == "ok"
        # Quota already used and circuit closed by the single success.
        assert cb.state() == State.CLOSED.value

    def test_extra_trial_calls_rejected_while_quota_pending(self):
        # half_open_max_calls=2, but we don't resolve the first call's
        # outcome before checking that a 3rd call would be rejected once
        # the quota of "calls made" (not yet resolved) is reached.
        cb = self._open_then_wait(half_open_max_calls=1)
        # Manually push it back to HALF_OPEN with a higher quota to test
        # the "quota reached" rejection path distinctly from closing.
        cb.half_open_max_calls = 2
        cb._state = State.HALF_OPEN
        cb._half_open_calls_made = 2
        cb._half_open_successes = 0
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(ok)


class TestCallSemantics:
    def test_call_passes_args_and_kwargs_through(self):
        cb = CircuitBreaker()

        def add(a, b, c=0):
            return a + b + c

        assert cb.call(add, 2, 3, c=4) == 9

    def test_original_exception_propagates(self):
        cb = CircuitBreaker(failure_threshold=5)

        def custom_fail():
            raise ValueError("specific error")

        with pytest.raises(ValueError, match="specific error"):
            cb.call(custom_fail)