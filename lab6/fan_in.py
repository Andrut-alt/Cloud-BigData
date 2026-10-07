"""A simple Fan-In (multiplexer) implementation.

Idea: merge several input streams into a single output stream, as items
become available from each of them, without one slow input blocking the
items already waiting on another input.

Each input is consumed on its own background thread, so a slow input
can never hold up items from a faster one (non-blocking merge). Every
item reaching the output is wrapped in Ok(value, source) or, if pulling
from an input raised an exception, Err(error, source) -- a minimal
Result<T, E> chosen so that one input's failure never stops the others:
that input's thread reports the error as its last item and finishes,
while the remaining inputs keep flowing. The output stream closes once
every input has finished (successfully or with an error).

Per-input order is preserved (items from the same input arrive in the
order that input produced them); the interleaving between different
inputs is not guaranteed and depends on timing.
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Union


@dataclass(frozen=True)
class Ok:
    """A value produced by input number `source`."""
    value: Any
    source: int


@dataclass(frozen=True)
class Err:
    """An error raised while pulling from input number `source`. That
    input stops after this; the other inputs are unaffected."""
    error: BaseException
    source: int


Result = Union[Ok, Err]

_CLOSE = object()  # internal sentinel: every input has finished


def fan_in(inputs: List[Iterable[Any]]) -> Iterator[Result]:
    """Merge several input streams into one, as items become available.

    :param inputs: a list of iterables (Stream<T>). Each one is read on
        its own thread, so a slow input never delays items from a
        faster one.
    :returns: an iterator of Ok/Err items. It stops (raises
        StopIteration, i.e. the `for` loop ends) once every input is
        exhausted or has failed -- this is how the output stream closes.
    """
    inputs = list(inputs)
    out = queue.Queue()
    remaining = len(inputs)
    lock = threading.Lock()

    def pump(source, stream):
        nonlocal remaining
        try:
            for value in stream:
                out.put(Ok(value, source))
        except BaseException as exc:  # one input's failure...
            out.put(Err(exc, source))  # ...becomes its last item...
        finally:
            with lock:
                remaining -= 1
                last_one = remaining == 0
            if last_one:
                out.put(_CLOSE)  # ...and never blocks the others.

    for source, stream in enumerate(inputs):
        threading.Thread(target=pump, args=(source, stream),
                         daemon=True).start()

    if not inputs:
        return

    while True:
        item = out.get()
        if item is _CLOSE:
            return
        yield item
