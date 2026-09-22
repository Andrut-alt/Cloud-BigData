"""Demo: watch a CircuitBreaker react to a flaky remote service.

Simulates a service that fails for a while, then recovers. Run this file
directly (not with pytest) to see the state transitions printed live:

    python3 demo.py
"""

import time

from circuit_breaker import (
    CallTimeoutError,
    CircuitBreaker,
    CircuitBreakerOpenError,
)


class FlakyService:
    """A fake remote service: fails on the first `fail_calls` calls,
    then starts succeeding (simulating a recovered dependency)."""

    def __init__(self, fail_calls):
        self.fail_calls = fail_calls
        self.call_count = 0

    def request(self):
        self.call_count += 1
        if self.call_count <= self.fail_calls:
            raise ConnectionError("service unavailable")
        return f"response #{self.call_count}"


def run_call(breaker, service, label):
    """Make one call through the breaker and print the outcome."""
    state_before = breaker.state()
    try:
        result = breaker.call(service.request)
        outcome = f"OK -> {result}"
    except CircuitBreakerOpenError:
        outcome = "BLOCKED (circuit is open)"
    except CallTimeoutError:
        outcome = "TIMEOUT"
    except ConnectionError:
        outcome = "FAILED (service error)"

    state_after = breaker.state()
    arrow = f"{state_before} -> {state_after}" if state_before != state_after \
        else state_before
    print(f"[{label:>2}] state: {arrow:<24} result: {outcome}")


def main():
    service = FlakyService(fail_calls=3)
    breaker = CircuitBreaker(
        failure_threshold=3,
        half_open_max_calls=2,
        open_state_duration=1.0,
        timeout_per_call=2.0,
    )

    print("=== Phase 1: service is down, breaker should trip to OPEN ===")
    for i in range(1, 5):
        run_call(breaker, service, i)

    print("\n=== Phase 2: calls blocked while OPEN ===")
    for i in range(5, 7):
        run_call(breaker, service, i)

    print(f"\nWaiting {breaker.open_state_duration}s for open_state_duration to pass...")
    time.sleep(breaker.open_state_duration + 0.1)

    print("\n=== Phase 3: HALF_OPEN trial calls (service has recovered) ===")
    for i in range(7, 10):
        run_call(breaker, service, i)

    print(f"\nFinal state: {breaker.state()}")


if __name__ == "__main__":
    main()