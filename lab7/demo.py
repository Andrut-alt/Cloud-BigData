"""Demo: watch Fan-Out spread artificially slow tasks across several
workers, compare that to a single worker, and see a graceful shutdown.

Run this file directly (not with pytest):

    python3 demo.py
"""

import time

from fan_out import Err, Ok, fan_out, run_to_completion

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


def slow_task(n):
    """Pretends to do 0.2s of work, occasionally failing."""
    time.sleep(0.2)
    if n == 5:
        raise ValueError(f"task {n} failed")
    return n * n


def run_and_time(workers, tasks):
    start = time.monotonic()
    items = run_to_completion(slow_task, workers=workers, tasks=tasks)
    elapsed = time.monotonic() - start
    oks = sum(1 for it in items if isinstance(it, Ok))
    errs = sum(1 for it in items if isinstance(it, Err))
    log(f"workers={workers}: {len(items)} results "
        f"({oks} ok, {errs} failed) in {elapsed:.2f}s")
    return elapsed


def main():
    tasks = list(range(8))

    print("=== Comparing 1 worker vs 4 workers on 8 slow tasks (0.2s each) ===")
    serial = run_and_time(workers=1, tasks=tasks)
    parallel = run_and_time(workers=4, tasks=tasks)
    log(f"speed-up: {serial / parallel:.1f}x")

    print("\n=== Streaming results as they complete, then graceful shutdown ===")
    fo = fan_out(slow_task, workers=3)
    for n in tasks:
        fo.submit(n)
    fo.shutdown()  # no more tasks accepted; the 8 already queued still run
    try:
        fo.submit(99)
    except Exception as exc:
        log(f"submit() after shutdown rejected: {exc!r}")

    for item in fo.results():
        if isinstance(item, Ok):
            log(f"  Ok  task={item.task} -> {item.value}")
        else:
            log(f"  Err task={item.task} -> {item.error!r}")
    log("results() closed: every queued task was accounted for")


if __name__ == "__main__":
    main()
