"""Demo: watch a Debouncer collapse a burst of events (e.g. typing).

Run this file directly (not with pytest):

    python3 demo.py
"""

import time

from debounce import Debouncer

START = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - START:5.2f}s] {msg}")


def search(query):
    log(f"  >> search executed for {query!r}")


def type_text(wrapper, text, pause):
    for i in range(1, len(text) + 1):
        log(f"typed {text[:i]!r}")
        wrapper(text[:i])
        time.sleep(pause)


def main():
    print("=== Trailing (default): search runs once, after typing stops ===")
    d = Debouncer(delay_ms=300).debounce(search)
    type_text(d, "hello", pause=0.1)
    time.sleep(0.6)

    print("\n=== Leading only: search runs on the first keystroke ===")
    d = Debouncer(delay_ms=300, leading=True, trailing=False).debounce(search)
    type_text(d, "world", pause=0.1)
    time.sleep(0.6)

    print("\n=== Leading + trailing: first and last keystroke ===")
    d = Debouncer(delay_ms=300, leading=True, trailing=True).debounce(search)
    type_text(d, "debounce", pause=0.1)
    time.sleep(0.6)

    print("\n=== dispose(): pending call is cancelled ===")
    d = Debouncer(delay_ms=300).debounce(search)
    type_text(d, "cancel", pause=0.05)
    log("dispose()")
    d.dispose()
    time.sleep(0.6)
    log("done, search was never executed")


if __name__ == "__main__":
    main()
