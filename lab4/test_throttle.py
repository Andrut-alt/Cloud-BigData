"""Tests for the Throttler class.

Window/limit scenarios use a fake clock, so they are instant and exact.
Scenarios that need the internal timer (trailing, queue) use short real
delays with time.sleep.
"""

import time

import pytest

from throttle import DROP, QUEUE, Throttler


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Recorder:
    def __init__(self):
        self.args = []
        self.times = []

    def __call__(self, value=None):
        self.args.append(value)
        self.times.append(time.monotonic())


class TestWindowLimit:
    def test_10_calls_in_one_second_with_limit_3_runs_only_3(self):
        clock = FakeClock()
        rec = Recorder()
        t = Throttler(rate=(3, 1.0), clock=clock).throttle(rec)
        for i in range(10):
            t(i)
            clock.advance(0.1 / 10)  # all 10 calls within ~0.1s
        assert rec.args == [0, 1, 2]

    def test_calls_spread_over_one_second_respect_the_rate(self):
        clock = FakeClock()
        rec = Recorder()
        thr = Throttler(rate=(3, 1.0), clock=clock)
        t = thr.throttle(rec)
        for i in range(10):  # 10 calls at 0.0, 0.1, ... 0.9 s
            t(i)
            clock.advance(0.1)
        # 3 from the initial burst + 2.7 tokens refilled over 0.9 s
        assert rec.args == [0, 1, 2, 4, 7]
        assert thr.dropped == 5

    def test_tokens_refill_after_a_pause(self):
        clock = FakeClock()
        rec = Recorder()
        t = Throttler(rate=(3, 1.0), clock=clock).throttle(rec)
        for i in range(5):
            t(i)
        assert rec.args == [0, 1, 2]
        clock.advance(1.0)
        for i in range(10, 15):
            t(i)
        assert rec.args == [0, 1, 2, 10, 11, 12]

    def test_bucket_never_exceeds_capacity(self):
        clock = FakeClock()
        rec = Recorder()
        t = Throttler(capacity=2, refill_rate=1, clock=clock).throttle(rec)
        clock.advance(100)  # long idle must not bank 100 tokens
        for i in range(5):
            t(i)
        assert rec.args == [0, 1]

    def test_token_bucket_parameters(self):
        clock = FakeClock()
        rec = Recorder()
        t = Throttler(capacity=5, refill_rate=2, clock=clock).throttle(rec)
        for i in range(8):
            t(i)
        assert len(rec.args) == 5
        clock.advance(1.0)  # 2 tokens/s -> 2 more calls
        for i in range(8):
            t(i)
        assert len(rec.args) == 7


class TestDropMode:
    def test_excess_calls_are_discarded(self):
        clock = FakeClock()
        rec = Recorder()
        thr = Throttler(rate=(2, 1.0), mode=DROP, clock=clock)
        t = thr.throttle(rec)
        for i in range(6):
            t(i)
        time.sleep(0.05)
        assert rec.args == [0, 1]
        assert thr.dropped == 4
        assert thr.executed == 2

    def test_discarded_calls_are_not_run_later(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, mode=DROP).throttle(rec)
        t("a")
        t("b")
        time.sleep(0.2)
        assert rec.args == ["a"]


class TestLeadingTrailing:
    def test_leading_runs_first_call_immediately(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=10, leading=True,
                      trailing=False).throttle(rec)
        t("first")
        assert rec.args == ["first"]

    def test_leading_only_skips_the_last_rejected_call(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, leading=True,
                      trailing=False).throttle(rec)
        t(1)
        t(2)
        t(3)
        time.sleep(0.2)
        assert rec.args == [1]

    def test_leading_and_trailing_runs_first_and_last(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, leading=True,
                      trailing=True).throttle(rec)
        t(1)
        t(2)
        t(3)
        assert rec.args == [1]
        time.sleep(0.2)
        assert rec.args == [1, 3]

    def test_trailing_only_runs_nothing_immediately(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, leading=False,
                      trailing=True).throttle(rec)
        t(1)
        t(2)
        t(3)
        assert rec.args == []
        time.sleep(0.2)
        assert rec.args == [3]

    def test_drop_mode_needs_leading_or_trailing(self):
        with pytest.raises(ValueError):
            Throttler(rate=(1, 1.0), leading=False, trailing=False)


class TestQueueMode:
    def test_order_is_preserved(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=50, mode=QUEUE).throttle(rec)
        for i in range(6):
            t(i)
        time.sleep(0.5)
        assert rec.args == [0, 1, 2, 3, 4, 5]

    def test_nothing_is_lost(self):
        rec = Recorder()
        thr = Throttler(capacity=2, refill_rate=50, mode=QUEUE)
        t = thr.throttle(rec)
        for i in range(10):
            t(i)
        time.sleep(0.6)
        assert rec.args == list(range(10))
        assert thr.dropped == 0

    def test_queued_calls_are_spaced_by_the_rate(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, mode=QUEUE).throttle(rec)
        for i in range(4):
            t(i)
        time.sleep(0.5)
        gaps = [b - a for a, b in zip(rec.times, rec.times[1:])]
        assert len(gaps) == 3
        assert all(gap >= 0.04 for gap in gaps)  # 1/20 s minus tolerance

    def test_queue_does_not_run_more_than_capacity_at_once(self):
        rec = Recorder()
        t = Throttler(capacity=3, refill_rate=10, mode=QUEUE).throttle(rec)
        for i in range(8):
            t(i)
        assert rec.args == [0, 1, 2]  # the rest waits for tokens
        time.sleep(1.0)
        assert rec.args == list(range(8))


class TestDispose:
    def test_dispose_discards_queued_calls(self):
        rec = Recorder()
        t = Throttler(capacity=1, refill_rate=20, mode=QUEUE).throttle(rec)
        for i in range(5):
            t(i)
        t.dispose()
        time.sleep(0.4)
        assert rec.args == [0]

    def test_calls_after_dispose_are_ignored(self):
        rec = Recorder()
        t = Throttler(rate=(5, 1.0)).throttle(rec)
        t.dispose()
        t(1)
        assert rec.args == []


class TestConfiguration:
    def test_rejects_missing_limit(self):
        with pytest.raises(ValueError):
            Throttler()

    def test_rejects_both_rate_and_bucket(self):
        with pytest.raises(ValueError):
            Throttler(rate=(3, 1.0), capacity=3, refill_rate=3)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            Throttler(rate=(3, 1.0), mode="skip")
