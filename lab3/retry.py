"""A simple Retry implementation.

Idea: safely repeat an operation that may fail temporarily. Between
attempts we wait according to a back-off strategy, and we give up early on
errors that are not worth retrying.

Back-off strategies (delay before retry number n, n = 1, 2, ...):
    constant           - base_delay
    exponential        - base_delay * 2 ** (n - 1)
    exponentialJitter  - random value in [0, base_delay * 2 ** (n - 1)]
                         ("full jitter", spreads out simultaneous clients)
"""

import random
import time

CONSTANT = "constant"
EXPONENTIAL = "exponential"
EXPONENTIAL_JITTER = "exponentialJitter"
STRATEGIES = (CONSTANT, EXPONENTIAL, EXPONENTIAL_JITTER)


class RetryExhaustedError(Exception):
    """Raised when every attempt failed with a retryable error.

    :ivar attempts: how many attempts were made.
    :ivar last_error: the exception raised by the final attempt.
    """

    def __init__(self, attempts, last_error):
        super().__init__(f"Gave up after {attempts} attempts: {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


class Retry:
    """Runs an operation until it succeeds or attempts run out.

    :param max_attempts: total number of calls to fn (first try included).
    :param strategy: "constant", "exponential" or "exponentialJitter".
    :param base_delay: base pause in seconds.
    :param retry_on: which errors are retryable. Either a tuple of exception
        classes, or a predicate ``exc -> bool`` (e.g. to check an error code).
        Errors that do not match are re-raised immediately.
    :param sleep: function used to wait (injectable for fast tests).
    :param rng: function returning a float in [0, 1) (injectable for tests).
    """

    def __init__(self, max_attempts=3, strategy=CONSTANT, base_delay=0.1,
                 retry_on=(Exception,), sleep=time.sleep, rng=random.random):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.base_delay = base_delay
        self.retry_on = retry_on
        self._sleep = sleep
        self._rng = rng

        # Pauses actually taken during the most recent execute() call.
        self.delays = []

    def execute(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs), retrying on retryable errors.

        :raises RetryExhaustedError: all max_attempts attempts failed.
        Non-retryable exceptions are re-raised as-is, without retrying.
        """
        self.delays = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                if attempt == self.max_attempts:
                    raise RetryExhaustedError(attempt, exc) from exc
            delay = self._delay_before_retry(attempt)
            self.delays.append(delay)
            self._sleep(delay)

    def _is_retryable(self, exc):
        if callable(self.retry_on) and not isinstance(self.retry_on, type):
            return bool(self.retry_on(exc))
        return isinstance(exc, self.retry_on)

    def _delay_before_retry(self, attempt):
        """Pause after the given (failed) attempt number, 1-based."""
        if self.strategy == CONSTANT:
            return self.base_delay
        cap = self.base_delay * 2 ** (attempt - 1)
        if self.strategy == EXPONENTIAL:
            return cap
        return self._rng() * cap
