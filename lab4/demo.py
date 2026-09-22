"""Demo: watch a Throttler limit a burst of calls to 3 per second.

Run this file directly (not with pytest):

    python3 demo.py
"""

import time

from throttle import DROP, QUEUE, Throttler

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


def send(n):
    log(f"  >> request #{n} executed")


def burst(wrapper, count, pause):
    for n in range(1, count + 1):
        log(f"call #{n}")
        wrapper(n)
        time.sleep(pause)


def main():
    print("=== Drop: limit 3/s, 10 calls every 0.1s, extra calls discarded ===")
    thr = Throttler(rate=(3, 1.0), mode=DROP)
    burst(thr.throttle(send), 10, 0.1)
    log(f"executed={thr.executed}, dropped={thr.dropped}")

    print("\n=== Drop + trailing: the last rejected call runs later ===")
    thr = Throttler(rate=(3, 1.0), mode=DROP, trailing=True)
    burst(thr.throttle(send), 5, 0.02)
    time.sleep(1.0)
    log(f"executed={thr.executed}, dropped={thr.dropped}")

    print("\n=== Queue: limit 3/s, nothing lost, order preserved ===")
    thr = Throttler(rate=(3, 1.0), mode=QUEUE)
    burst(thr.throttle(send), 6, 0.02)
    time.sleep(1.5)
    log(f"executed={thr.executed}, dropped={thr.dropped}")


if __name__ == "__main__":
    main()
