"""Tests for the Retry class.

sleep is replaced with a recorder, so the tests run instantly and can
assert the exact pauses (and their sum) instead of measuring real time.
"""

import pytest

from retry import (
    CONSTANT,
    EXPONENTIAL,
    EXPONENTIAL_JITTER,
    Retry,
    RetryExhaustedError,
)


class FlakyFn:
    """Mock fn: raises `error` for the first `fail_times` calls, then
    returns "ok"."""

    def __init__(self, fail_times, error=ConnectionError("temporary")):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return "ok"


class Sleeper:
    """Records requested pauses instead of actually sleeping."""

    def __init__(self):
        self.pauses = []

    def __call__(self, seconds):
        self.pauses.append(seconds)


def make(sleeper=None, **kwargs):
    return Retry(sleep=sleeper or Sleeper(), **kwargs)


class TestSuccess:
    def test_success_on_first_attempt_no_retry_no_sleep(self):
        sleeper = Sleeper()
        fn = FlakyFn(fail_times=0)
        assert make(sleeper, max_attempts=3).execute(fn) == "ok"
        assert fn.calls == 1
        assert sleeper.pauses == []

    @pytest.mark.parametrize("fail_times", [1, 2, 4])
    def test_success_on_nth_attempt(self, fail_times):
        sleeper = Sleeper()
        fn = FlakyFn(fail_times=fail_times)
        r = make(sleeper, max_attempts=5, base_delay=1)
        assert r.execute(fn) == "ok"
        assert fn.calls == fail_times + 1
        assert len(sleeper.pauses) == fail_times

    def test_success_on_last_allowed_attempt(self):
        fn = FlakyFn(fail_times=2)
        assert make(max_attempts=3).execute(fn) == "ok"
        assert fn.calls == 3

    def test_args_and_kwargs_are_passed_through(self):
        seen = []
        make().execute(lambda *a, **k: seen.append((a, k)), 1, 2, x=3)
        assert seen == [((1, 2), {"x": 3})]


class TestExhaustion:
    def test_stops_after_max_attempts(self):
        fn = FlakyFn(fail_times=100)
        with pytest.raises(RetryExhaustedError) as info:
            make(max_attempts=4).execute(fn)
        assert fn.calls == 4
        assert info.value.attempts == 4
        assert info.value.last_error is fn.error

    def test_constant_delays_sum(self):
        sleeper = Sleeper()
        r = make(sleeper, max_attempts=4, strategy=CONSTANT, base_delay=0.5)
        with pytest.raises(RetryExhaustedError):
            r.execute(FlakyFn(fail_times=100))
        # 4 attempts -> 3 pauses, none after the final failure
        assert sleeper.pauses == [0.5, 0.5, 0.5]
        assert sum(sleeper.pauses) == pytest.approx(1.5)
        assert r.delays == sleeper.pauses

    def test_exponential_delays_sum(self):
        sleeper = Sleeper()
        r = make(sleeper, max_attempts=5, strategy=EXPONENTIAL, base_delay=1)
        with pytest.raises(RetryExhaustedError):
            r.execute(FlakyFn(fail_times=100))
        assert sleeper.pauses == [1, 2, 4, 8]
        assert sum(sleeper.pauses) == 15

    def test_exponential_jitter_stays_within_cap(self):
        sleeper = Sleeper()
        r = make(sleeper, max_attempts=5, strategy=EXPONENTIAL_JITTER,
                 base_delay=1)
        with pytest.raises(RetryExhaustedError):
            r.execute(FlakyFn(fail_times=100))
        caps = [1, 2, 4, 8]
        assert len(sleeper.pauses) == 4
        for pause, cap in zip(sleeper.pauses, caps):
            assert 0 <= pause <= cap

    def test_exponential_jitter_uses_rng(self):
        sleeper = Sleeper()
        r = Retry(max_attempts=4, strategy=EXPONENTIAL_JITTER, base_delay=1,
                  sleep=sleeper, rng=lambda: 0.5)
        with pytest.raises(RetryExhaustedError):
            r.execute(FlakyFn(fail_times=100))
        assert sleeper.pauses == [0.5, 1.0, 2.0]

    def test_max_attempts_one_never_sleeps(self):
        sleeper = Sleeper()
        with pytest.raises(RetryExhaustedError):
            make(sleeper, max_attempts=1).execute(FlakyFn(fail_times=1))
        assert sleeper.pauses == []


class TestRetryOn:
    def test_non_retryable_error_is_not_retried(self):
        sleeper = Sleeper()
        fn = FlakyFn(fail_times=100, error=ValueError("bad input"))
        r = make(sleeper, max_attempts=5, retry_on=(ConnectionError,))
        with pytest.raises(ValueError):
            r.execute(fn)
        assert fn.calls == 1
        assert sleeper.pauses == []

    def test_retryable_error_class_is_retried(self):
        fn = FlakyFn(fail_times=2, error=TimeoutError("slow"))
        r = make(max_attempts=5, retry_on=(ConnectionError, TimeoutError))
        assert r.execute(fn) == "ok"
        assert fn.calls == 3

    def test_subclass_of_retryable_error_is_retried(self):
        fn = FlakyFn(fail_times=1, error=ConnectionResetError("reset"))
        assert make(retry_on=(ConnectionError,)).execute(fn) == "ok"

    def test_predicate_can_select_by_error_code(self):
        class HttpError(Exception):
            def __init__(self, code):
                super().__init__(code)
                self.code = code

        def retryable(exc):
            return isinstance(exc, HttpError) and exc.code in (429, 503)

        # 503 -> retried, then success
        fn = FlakyFn(fail_times=2, error=HttpError(503))
        assert make(max_attempts=5, retry_on=retryable).execute(fn) == "ok"
        assert fn.calls == 3

        # 404 -> not retried
        fn = FlakyFn(fail_times=100, error=HttpError(404))
        with pytest.raises(HttpError):
            make(max_attempts=5, retry_on=retryable).execute(fn)
        assert fn.calls == 1

    def test_error_becoming_non_retryable_stops_immediately(self):
        errors = iter([ConnectionError("a"), ValueError("fatal")])

        def fn():
            raise next(errors)

        r = make(max_attempts=5, retry_on=(ConnectionError,))
        with pytest.raises(ValueError):
            r.execute(fn)


class TestConfiguration:
    def test_rejects_zero_attempts(self):
        with pytest.raises(ValueError):
            Retry(max_attempts=0)

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError):
            Retry(strategy="linear")

    def test_delays_reset_between_executions(self):
        sleeper = Sleeper()
        r = make(sleeper, max_attempts=3, base_delay=1)
        r.execute(FlakyFn(fail_times=2))
        assert r.delays == [1, 1]
        r.execute(FlakyFn(fail_times=0))
        assert r.delays == []
