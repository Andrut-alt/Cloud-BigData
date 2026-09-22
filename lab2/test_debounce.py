"""Tests for the Debouncer class.

Time-dependent scenarios use small real delays with time.sleep, keeping the
suite fast while exercising real timer behaviour.
"""

import time

import pytest

from debounce import Debouncer

DELAY = 100  # ms
WAIT = 0.25  # s, comfortably longer than DELAY


class Recorder:
    """Collects the arguments of every fn invocation."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestTrailingDefault:
    def test_burst_of_calls_runs_fn_once(self):
        rec = Recorder()
        d = Debouncer(DELAY).debounce(rec)
        for i in range(5):
            d(i)
            time.sleep(0.02)
        assert rec.calls == []
        time.sleep(WAIT)
        assert len(rec.calls) == 1

    def test_uses_arguments_of_last_call(self):
        rec = Recorder()
        d = Debouncer(DELAY).debounce(rec)
        d(1, key="a")
        d(2, key="b")
        d(3, key="c")
        time.sleep(WAIT)
        assert rec.calls == [((3,), {"key": "c"})]

    def test_each_call_restarts_the_timer(self):
        rec = Recorder()
        d = Debouncer(DELAY).debounce(rec)
        for _ in range(4):
            d()
            time.sleep(0.07)  # < DELAY, so the timer keeps restarting
        assert rec.calls == []
        time.sleep(WAIT)
        assert len(rec.calls) == 1

    def test_separate_bursts_run_separately(self):
        rec = Recorder()
        d = Debouncer(DELAY).debounce(rec)
        d("first")
        time.sleep(WAIT)
        d("second")
        time.sleep(WAIT)
        assert rec.calls == [(("first",), {}), (("second",), {})]


class TestLeading:
    def test_first_call_runs_immediately(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True, trailing=False).debounce(rec)
        d("x")
        assert rec.calls == [(("x",), {})]

    def test_subsequent_calls_suppressed_until_pause(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True, trailing=False).debounce(rec)
        d(1)
        d(2)
        d(3)
        time.sleep(WAIT)
        assert rec.calls == [((1,), {})]

    def test_runs_again_after_pause(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True, trailing=False).debounce(rec)
        d(1)
        d(2)
        time.sleep(WAIT)
        d(3)
        assert rec.calls == [((1,), {}), ((3,), {})]


class TestLeadingAndTrailing:
    def test_runs_first_and_last_of_a_burst(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True, trailing=True).debounce(rec)
        d(1)
        d(2)
        d(3)
        assert rec.calls == [((1,), {})]
        time.sleep(WAIT)
        assert rec.calls == [((1,), {}), ((3,), {})]

    def test_single_call_runs_only_once(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True, trailing=True).debounce(rec)
        d(1)
        time.sleep(WAIT)
        assert rec.calls == [((1,), {})]


class TestOptions:
    def test_both_disabled_is_rejected(self):
        with pytest.raises(ValueError):
            Debouncer(DELAY, leading=False, trailing=False)


class TestDispose:
    def test_dispose_cancels_pending_trailing_call(self):
        rec = Recorder()
        d = Debouncer(DELAY).debounce(rec)
        d(1)
        d.dispose()
        time.sleep(WAIT)
        assert rec.calls == []

    def test_calls_after_dispose_are_ignored(self):
        rec = Recorder()
        d = Debouncer(DELAY, leading=True).debounce(rec)
        d.dispose()
        d(1)
        time.sleep(WAIT)
        assert rec.calls == []

    def test_dispose_is_idempotent(self):
        d = Debouncer(DELAY).debounce(Recorder())
        d.dispose()
        d.dispose()
