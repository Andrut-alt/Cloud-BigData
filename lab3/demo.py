"""Demo: watch Retry recover from a temporarily failing service.

Run this file directly (not with pytest):

    python3 demo.py
"""

import time

from retry import (
    CONSTANT,
    EXPONENTIAL,
    EXPONENTIAL_JITTER,
    Retry,
    RetryExhaustedError,
)

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


class FlakyService:
    """Fails with `error` on the first `fail_calls` calls, then succeeds."""

    def __init__(self, fail_calls, error=ConnectionError("service unavailable")):
        self.fail_calls = fail_calls
        self.error = error
        self.call_count = 0

    def request(self):
        self.call_count += 1
        if self.call_count <= self.fail_calls:
            log(f"  attempt {self.call_count}: FAILED ({self.error})")
            raise self.error
        log(f"  attempt {self.call_count}: OK")
        return f"response #{self.call_count}"


def run(title, retry, service):
    print(f"\n=== {title} ===")
    try:
        result = retry.execute(service.request)
        log(f"result: {result}")
    except RetryExhaustedError as exc:
        log(f"GAVE UP after {exc.attempts} attempts")
    except Exception as exc:
        log(f"NOT RETRIED, raised immediately: {exc!r}")
    print(f"    pauses: {[round(d, 2) for d in retry.delays]}")


def main():
    run("constant: service recovers on 3rd attempt",
        Retry(max_attempts=5, strategy=CONSTANT, base_delay=0.2,
              retry_on=(ConnectionError,)),
        FlakyService(fail_calls=2))

    run("exponential: pauses double each time",
        Retry(max_attempts=4, strategy=EXPONENTIAL, base_delay=0.1,
              retry_on=(ConnectionError,)),
        FlakyService(fail_calls=2))

    run("exponentialJitter: random pause up to the exponential cap",
        Retry(max_attempts=4, strategy=EXPONENTIAL_JITTER, base_delay=0.2,
              retry_on=(ConnectionError,)),
        FlakyService(fail_calls=2))

    run("attempts exhausted: service never recovers",
        Retry(max_attempts=3, strategy=EXPONENTIAL, base_delay=0.1,
              retry_on=(ConnectionError,)),
        FlakyService(fail_calls=100))

    run("non-retryable error (ValueError) is raised at once",
        Retry(max_attempts=5, strategy=CONSTANT, base_delay=0.1,
              retry_on=(ConnectionError,)),
        FlakyService(fail_calls=100, error=ValueError("bad request")))


if __name__ == "__main__":
    main()
