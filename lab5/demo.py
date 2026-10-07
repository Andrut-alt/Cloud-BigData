"""Demo: watch with_timeout cap how long an operation may run, and see it
combined with Retry so the total wait stays bounded.

Run this file directly (not with pytest):

    python3 demo.py
"""

import os
import sys
import time

from timeout import OperationTimeoutError, Timeout, with_timeout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab3"))
from retry import CONSTANT, Retry, RetryExhaustedError  # noqa: E402

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


def fast_query():
    time.sleep(0.05)
    return "fast result"


def slow_query(cancel_event):
    for _ in range(20):
        if cancel_event.is_set():
            log("  slow_query noticed cancellation, stopping early")
            return "abandoned"
        time.sleep(0.03)
    log("  slow_query: SIDE EFFECT (must never be printed)")
    return "too late"


class FlakySlowService:
    """The first `slow_calls` calls exceed the timeout; then it answers
    fast, as if the remote dependency had recovered."""

    def __init__(self, slow_calls):
        self.slow_calls = slow_calls
        self.call_count = 0

    def request(self, cancel_event):
        self.call_count += 1
        if self.call_count <= self.slow_calls:
            return slow_query(cancel_event)
        return fast_query()


def main():
    print("=== Operation finishes before the timeout ===")
    log(f"result: {with_timeout(fast_query, timeout_ms=200)}")

    print("\n=== Operation exceeds the timeout: raised, side effect cancelled ===")
    try:
        with_timeout(slow_query, timeout_ms=100)
    except OperationTimeoutError as exc:
        log(f"TIMEOUT: {exc}")
    time.sleep(0.5)  # let the abandoned worker thread notice cancel_event
    log("(no SIDE EFFECT line above it -> the late write never happened)")

    print("\n=== Timeout + Retry: bounded total wait, not multiplied endlessly ===")
    service = FlakySlowService(slow_calls=2)
    timeout = Timeout(timeout_ms=100)
    retry = Retry(max_attempts=4, strategy=CONSTANT, base_delay=0.05,
                  retry_on=(OperationTimeoutError,))
    try:
        result = retry.execute(timeout.run, service.request)
        log(f"result: {result}")
    except RetryExhaustedError as exc:
        log(f"GAVE UP after {exc.attempts} attempts")
    bound = retry.max_attempts * (timeout.timeout_ms / 1000) + \
        retry.max_attempts * retry.base_delay
    log(f"worst-case bound was max_attempts*(timeout+base_delay) "
        f"= {bound:.2f}s -- it does not grow without limit")


if __name__ == "__main__":
    main()
