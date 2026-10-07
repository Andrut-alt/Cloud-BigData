"""Tests for FanOut / fan_out().

Each test that drains results() does so via collect(), which runs the
consumption on a background thread with a timeout -- a FanOut that
forgot to close would show up as a test failure, not a hung suite.
"""

import threading
import time
from collections import Counter

import pytest

from fan_out import Err, FanOut, FanOutClosedError, fan_out, run_to_completion


def collect(fo, timeout=2.0):
    """Drain fo.results() on a background thread; fails the test if
    results() never closes."""
    items = []

    def run():
        for item in fo.results():
            items.append(item)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "results() did not close after shutdown"
    return items


class TestBasicUsage:
    def test_submit_and_results_roundtrip(self):
        fo = fan_out(lambda x: x * 2, workers=3)
        for i in range(10):
            fo.submit(i)
        fo.shutdown()
        items = collect(fo)
        values = sorted(it.value for it in items)
        assert values == [i * 2 for i in range(10)]

    def test_run_to_completion_helper(self):
        items = run_to_completion(lambda x: x + 1, workers=4, tasks=range(5))
        assert sorted(it.value for it in items) == [1, 2, 3, 4, 5]

    def test_rejects_non_positive_worker_count(self):
        with pytest.raises(ValueError):
            FanOut(lambda x: x, workers=0)


class TestDataSafety:
    def test_every_task_is_processed_exactly_once(self):
        """No races, no double processing: with many workers racing for
        the same queue, each of N tasks must appear in the results
        exactly once, with the correct value."""
        tasks = list(range(300))
        fo = fan_out(lambda x: x, workers=8)
        for t in tasks:
            fo.submit(t)
        fo.shutdown()
        items = collect(fo)

        task_counts = Counter(it.task for it in items)
        assert len(items) == len(tasks)
        assert set(task_counts) == set(tasks)
        assert all(count == 1 for count in task_counts.values())
        assert all(it.value == it.task for it in items)  # process(x) = x
        assert fo.completed == len(tasks)

    def test_shared_counter_updates_are_not_lost(self):
        """A classic race-condition probe: every worker increments a
        plain (unlocked) Python int via the GIL-serialized bytecode for
        `+= 1` inside process(); FanOut itself adds no extra races, so
        with the queue handing out each task once, the final count must
        match exactly."""
        counter = {"value": 0}
        lock = threading.Lock()

        def process(_task):
            with lock:
                counter["value"] += 1
            return counter["value"]

        fo = fan_out(process, workers=10)
        for i in range(500):
            fo.submit(i)
        fo.shutdown()
        items = collect(fo)
        assert len(items) == 500
        assert counter["value"] == 500


class TestParallelism:
    def test_many_workers_are_faster_than_one_for_slow_tasks(self):
        def slow(_task):
            time.sleep(0.05)
            return "done"

        tasks = list(range(8))

        start = time.monotonic()
        run_to_completion(slow, workers=1, tasks=tasks)
        serial = time.monotonic() - start

        start = time.monotonic()
        run_to_completion(slow, workers=4, tasks=tasks)
        parallel = time.monotonic() - start

        # 8 tasks * 0.05s serially is ~0.4s; with 4 workers it should be
        # close to 2 batches * 0.05s =~ 0.1s. A generous factor keeps
        # this robust on a loaded CI machine while still proving real
        # concurrency, not just measurement noise.
        assert parallel < serial / 2


class TestGracefulShutdown:
    def test_already_queued_tasks_finish_after_shutdown(self):
        fo = fan_out(lambda x: x, workers=2)
        for i in range(20):
            fo.submit(i)
        fo.shutdown()  # called immediately, before workers drain the backlog
        items = collect(fo)
        assert sorted(it.value for it in items) == list(range(20))

    def test_submit_after_shutdown_is_rejected(self):
        fo = fan_out(lambda x: x, workers=2)
        fo.submit(1)
        fo.shutdown()
        with pytest.raises(FanOutClosedError):
            fo.submit(2)
        collect(fo)  # drain so the worker thread is not left hanging

    def test_results_closes_with_no_tasks_submitted(self):
        fo = fan_out(lambda x: x, workers=3)
        fo.shutdown()
        assert collect(fo) == []

    def test_shutdown_is_idempotent(self):
        fo = fan_out(lambda x: x, workers=2)
        fo.submit(1)
        fo.shutdown()
        fo.shutdown()  # must not raise or hang results()
        items = collect(fo)
        assert len(items) == 1


class TestErrorHandling:
    def test_a_failing_task_does_not_crash_its_worker(self):
        def process(x):
            if x % 2 == 0:
                raise ValueError(f"bad task {x}")
            return x * 10

        items = run_to_completion(process, workers=3, tasks=range(6))
        oks = {it.task: it.value for it in items if not isinstance(it, Err)}
        errs = {it.task for it in items if isinstance(it, Err)}

        assert oks == {1: 10, 3: 30, 5: 50}
        assert errs == {0, 2, 4}
        assert len(items) == 6  # nothing lost because one task failed
