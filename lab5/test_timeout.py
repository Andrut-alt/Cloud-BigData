"""Tests for with_timeout() / Timeout.

The combination tests reuse the Retry class from ../lab3, which is the
companion pattern most likely to be chained with a timeout in practice.
"""

import os
import sys
import threading
import time

import pytest

from timeout import OperationTimeoutError, Timeout, with_timeout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab3"))
from retry import CONSTANT, Retry, RetryExhaustedError  # noqa: E402


class TestSuccess:
    def test_operation_finishing_before_timeout_returns_its_result(self):
        assert with_timeout(lambda: 42, timeout_ms=200) == 42

    def test_arguments_are_passed_through(self):
        assert with_timeout(lambda a, b, c=0: a + b + c, 200, 2, 3, c=4) == 9

    def test_result_arrives_right_away_not_after_the_full_timeout(self):
        start = time.monotonic()
        with_timeout(lambda: "fast", timeout_ms=5000)
        assert time.monotonic() - start < 0.5

    def test_exception_raised_within_deadline_propagates_as_is(self):
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            with_timeout(fail, timeout_ms=200)


class TestTimeoutExceeded:
    def test_slow_operation_raises_operation_timeout_error(self):
        with pytest.raises(OperationTimeoutError):
            with_timeout(lambda: time.sleep(1), timeout_ms=50)

    def test_cooperative_side_effect_is_cancelled(self):
        side_effects = []
        cancelled = threading.Event()

        def slow(cancel_event):
            for _ in range(50):
                if cancel_event.is_set():
                    cancelled.set()
                    return "abandoned"
                time.sleep(0.02)
            side_effects.append("done")  # must never run
            return "finished"

        with pytest.raises(OperationTimeoutError):
            with_timeout(slow, timeout_ms=50)

        time.sleep(0.5)  # give the background thread a chance to notice
        assert cancelled.is_set()
        assert side_effects == []

    def test_uncooperative_function_is_not_passed_a_cancel_event(self):
        # fn has no cancel_event parameter -> with_timeout must not pass
        # one (it would otherwise blow up with an unexpected-kwarg error
        # instead of a clean OperationTimeoutError).
        def no_such_param(a, b):
            time.sleep(0.2)
            return a + b

        with pytest.raises(OperationTimeoutError):
            with_timeout(no_such_param, timeout_ms=50, a=1, b=2)


class TestTimeoutClass:
    def test_run_behaves_like_the_free_function(self):
        t = Timeout(timeout_ms=200)
        assert t.run(lambda: "ok") == "ok"
        with pytest.raises(OperationTimeoutError):
            t.run(lambda: time.sleep(1))


class TestCombinedWithRetry:
    """withTimeout composed with Retry: every attempt gets its own
    deadline, and the worst-case total wait stays bounded by
    max_attempts * timeout_ms (+ backoff) -- it is never open-ended."""

    def test_attempts_are_bounded_not_unbounded(self):
        calls = []

        def always_slow(cancel_event):
            calls.append(1)
            # Would block forever if with_timeout did not cancel it.
            cancel_event.wait(5)
            return "too late"

        timeout = Timeout(timeout_ms=30)
        retry = Retry(max_attempts=3, strategy=CONSTANT, base_delay=0.01,
                      retry_on=(OperationTimeoutError,))

        start = time.monotonic()
        with pytest.raises(RetryExhaustedError) as info:
            retry.execute(timeout.run, always_slow)
        elapsed = time.monotonic() - start

        assert len(calls) == 3  # exactly max_attempts, never more
        assert info.value.attempts == 3
        # Deterministic upper bound: 3 timeouts (30ms) + 2 backoffs (10ms).
        assert elapsed < 3 * 0.03 + 2 * 0.01 + 0.3  # + generous slack

    def test_succeeds_once_the_operation_becomes_fast_enough(self):
        attempt = [0]

        def sometimes_slow(cancel_event):
            attempt[0] += 1
            if attempt[0] < 3:
                cancel_event.wait(5)
                return "too late"
            return "ok"

        timeout = Timeout(timeout_ms=30)
        retry = Retry(max_attempts=5, strategy=CONSTANT, base_delay=0.01,
                      retry_on=(OperationTimeoutError,))

        assert retry.execute(timeout.run, sometimes_slow) == "ok"
        assert attempt[0] == 3

    def test_non_retryable_error_from_fn_still_propagates(self):
        def fails_fast(cancel_event):
            raise ValueError("not a timeout at all")

        timeout = Timeout(timeout_ms=100)
        retry = Retry(max_attempts=5, retry_on=(OperationTimeoutError,))

        with pytest.raises(ValueError):
            retry.execute(timeout.run, fails_fast)
