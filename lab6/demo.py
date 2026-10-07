"""Demo: watch fan_in() merge three independent, differently-paced
streams -- one of them failing partway through -- into a single output.

Run this file directly (not with pytest):

    python3 demo.py
"""

import time

from fan_in import Err, Ok, fan_in

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


def sensor(name, readings, delay):
    """A stream that reports one reading every `delay` seconds."""
    for value in readings:
        time.sleep(delay)
        yield f"{name}={value}"


def flaky_sensor(name, readings, delay, fail_after):
    """Like sensor(), but raises after `fail_after` readings -- simulates
    a sensor that disconnects partway through."""
    for i, value in enumerate(readings):
        time.sleep(delay)
        if i >= fail_after:
            raise ConnectionError(f"{name} disconnected")
        yield f"{name}={value}"


def main():
    print("=== Merging a fast sensor, a slow sensor, and one that fails ===")
    inputs = [
        sensor("fast", range(5), delay=0.08),
        sensor("slow", range(3), delay=0.25),
        flaky_sensor("flaky", range(5), delay=0.12, fail_after=2),
    ]

    names = {0: "fast", 1: "slow", 2: "flaky"}
    counts = {0: 0, 1: 0, 2: 0}
    errors = 0

    for item in fan_in(inputs):
        if isinstance(item, Ok):
            counts[item.source] += 1
            log(f"  Ok  from {names[item.source]}: {item.value}")
        else:
            errors += 1
            log(f"  Err from {names[item.source]}: {item.error!r}")

    log(f"stream closed -- all inputs finished "
        f"(counts={counts}, errors={errors})")
    print("\nNotice how 'fast' and 'flaky' items interleave freely while\n"
          "'slow' is still producing, and the flaky sensor's failure\n"
          "does not stop 'fast' or 'slow' from continuing to the end.")


if __name__ == "__main__":
    main()
