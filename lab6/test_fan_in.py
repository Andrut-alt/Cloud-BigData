"""Tests for fan_in().

Each test drains fan_in() from a background thread via collect(), so a
merge that forgot to close would show up as a test failure (thread still
alive after the timeout) instead of hanging the whole suite.
"""

import threading
import time

import pytest

from fan_in import Err, Ok, fan_in


def delayed(values, delay):
    """A stream that yields each value after a pause (simulates a slow
    source, e.g. a sensor or a network call)."""
    def generator():
        for v in values:
            time.sleep(delay)
            yield v
    return generator()


class FailingStream:
    """Yields `values`, then raises `error` instead of finishing."""

    def __init__(self, values, error):
        self.values = values
        self.error = error

    def __iter__(self):
        yield from self.values
        raise self.error


def collect(inputs, timeout=2.0):
    """Run fan_in(inputs) to completion on a background thread and
    return everything it produced; fails the test if it never closes."""
    results = []

    def run():
        for item in fan_in(inputs):
            results.append(item)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "fan_in did not close after all inputs finished"
    return results


class TestMerging:
    def test_merges_two_streams_preserving_per_stream_order(self):
        items = collect([iter([1, 2, 3]), iter(["a", "b", "c"])])
        by_source = {0: [], 1: []}
        for it in items:
            assert isinstance(it, Ok)
            by_source[it.source].append(it.value)
        assert by_source[0] == [1, 2, 3]
        assert by_source[1] == ["a", "b", "c"]

    def test_merges_three_streams_preserving_per_stream_order(self):
        items = collect([range(3), range(10, 13), range(100, 104)])
        by_source = {0: [], 1: [], 2: []}
        for it in items:
            by_source[it.source].append(it.value)
        assert by_source[0] == [0, 1, 2]
        assert by_source[1] == [10, 11, 12]
        assert by_source[2] == [100, 101, 102, 103]

    def test_global_interleaving_is_not_guaranteed_but_all_values_arrive(self):
        items = collect([range(5), range(5, 10)])
        assert sorted(it.value for it in items) == list(range(10))


class TestClosing:
    def test_closes_after_all_inputs_are_done(self):
        # collect() itself asserts the merge does not hang; here we only
        # check that every item was actually delivered before closing.
        items = collect([iter([1, 2]), iter([3])])
        assert len(items) == 3

    def test_closes_immediately_with_no_inputs(self):
        assert collect([]) == []

    def test_closes_after_a_single_input(self):
        assert collect([iter([1, 2, 3])]) == [Ok(1, 0), Ok(2, 0), Ok(3, 0)]


class TestErrorHandling:
    def test_error_in_one_input_does_not_stop_the_others(self):
        boom = ValueError("boom")
        items = collect([
            FailingStream([1, 2], boom),
            iter(["a", "b", "c"]),
        ])
        ok_from_second = [it.value for it in items
                           if isinstance(it, Ok) and it.source == 1]
        assert ok_from_second == ["a", "b", "c"]

        errors = [it for it in items if isinstance(it, Err)]
        assert len(errors) == 1
        assert errors[0].source == 0
        assert errors[0].error is boom

    def test_error_is_delivered_as_the_last_item_of_its_input(self):
        boom = RuntimeError("fail")
        items = collect([FailingStream([1, 2, 3], boom)])
        assert items == [Ok(1, 0), Ok(2, 0), Ok(3, 0), Err(boom, 0)]

    def test_output_still_closes_when_every_input_fails(self):
        e1, e2 = ValueError("one"), KeyError("two")
        items = collect([FailingStream([], e1), FailingStream([], e2)])
        assert all(isinstance(it, Err) for it in items)
        assert {it.source for it in items} == {0, 1}


class TestNonBlockingMerge:
    def test_fast_input_is_not_held_up_by_a_slow_one(self):
        start = time.monotonic()
        arrivals = []

        def run():
            inputs = [delayed([1, 2, 3], 0.3), iter(["a", "b", "c"])]
            for item in fan_in(inputs):
                arrivals.append((time.monotonic() - start, item))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(2.0)
        assert not worker.is_alive()

        fast_times = [t for t, item in arrivals if item.source == 1]
        slow_times = [t for t, item in arrivals if item.source == 0]
        # The fast stream's items all arrive almost immediately, well
        # before the slow stream even produces its first one (~0.3s) --
        # proving the merge does not wait on the slow input.
        assert max(fast_times) < 0.2
        assert min(slow_times) >= 0.28  # ~0.3s, with a small timing margin
