"""A simple Fan-Out (demultiplexer) implementation.

Idea: spread work coming from a single source across N worker threads so
independent tasks run concurrently instead of one after another.

Components:
    task queue     - a thread-safe queue.Queue(); submit() pushes tasks
                     onto it, workers pop from it. Queue.get()/put() are
                     internally synchronized, so a task is handed to
                     exactly one worker -- no two workers can ever pop
                     the same task (no races, no double processing).
    N workers      - daemon threads, each looping: pop a task, run
                     process(task), push the outcome onto the results
                     queue, repeat until told to stop.
    results queue  - aggregates every worker's outcome, wrapped in
                     Ok(task, value) or Err(task, error) so a result can
                     always be traced back to the task that produced it.

Graceful shutdown: shutdown() stops accepting new tasks but lets every
task already queued finish normally -- results() keeps yielding until
the queue is fully drained and every worker has exited, then closes
itself. No task that was submitted before shutdown() is ever dropped.
"""

import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any

_STOP = object()          # tells a worker to exit, once it reaches this
_RESULTS_DONE = object()  # tells results() every worker has exited


@dataclass(frozen=True)
class Ok:
    """process(task) returned `value` successfully."""
    task: Any
    value: Any


@dataclass(frozen=True)
class Err:
    """process(task) raised `error`; the worker that ran it kept going."""
    task: Any
    error: BaseException


class FanOutClosedError(Exception):
    """Raised by submit() after shutdown() has already been called."""


class FanOut:
    """Distributes tasks from a single submitter across `workers` worker
    threads running `process`, and aggregates their outcomes.

    :param process: callable Task -> Result, run on a worker thread.
    :param workers: number of worker threads (N).
    """

    def __init__(self, process, workers):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.process = process
        self.workers = workers

        self._tasks = Queue()
        self._results = Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.submitted = 0
        self.completed = 0

        self._threads = [threading.Thread(target=self._worker_loop, daemon=True)
                          for _ in range(workers)]
        for t in self._threads:
            t.start()

    def submit(self, task):
        """Queue a task for processing by one of the workers.

        :raises FanOutClosedError: shutdown() has already been called.
        """
        with self._lock:
            if self._closed:
                raise FanOutClosedError(
                    "FanOut is shutting down; no new tasks are accepted")
            self.submitted += 1
        self._tasks.put(task)

    def results(self):
        """Yield Ok(task, value) / Err(task, error) for every submitted
        task, in completion order (not submission order -- a slow task
        does not hold up a faster one that was queued after it).

        Stops once shutdown() has been called and every queued task has
        been processed; never discards a result that was produced.
        """
        while True:
            item = self._results.get()
            if item is _RESULTS_DONE:
                return
            yield item

    def shutdown(self):
        """Stop accepting new tasks (further submit() calls raise
        FanOutClosedError). Tasks already queued are still processed --
        that is the "graceful" part. Non-blocking: spawns a background
        watcher that waits for every worker to finish its backlog and
        only then closes results(). Calling shutdown() twice is safe.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for _ in range(self.workers):
            self._tasks.put(_STOP)

        def wait_then_close():
            for t in self._threads:
                t.join()
            self._results.put(_RESULTS_DONE)

        threading.Thread(target=wait_then_close, daemon=True).start()

    def _worker_loop(self):
        while True:
            task = self._tasks.get()
            if task is _STOP:
                return
            try:
                value = self.process(task)
                self._results.put(Ok(task, value))
            except Exception as exc:
                self._results.put(Err(task, exc))
            finally:
                with self._lock:
                    self.completed += 1


def fan_out(process, workers):
    """fanOut(process, workers) -> FanOut, exposing submit(task) and
    results()."""
    return FanOut(process, workers)


def run_to_completion(process, workers, tasks):
    """Optional results aggregator: submit every task, shut down once
    they are all queued, and return every Ok/Err result as a list."""
    fo = FanOut(process, workers)
    for task in tasks:
        fo.submit(task)
    fo.shutdown()
    return list(fo.results())
