# utils/async_utils.py
import asyncio
import queue
import threading
from typing import AsyncIterator, Callable, Iterable, TypeVar

T = TypeVar("T")
_SENTINEL = object()


async def iter_in_thread(fn: Callable[..., Iterable[T]], *args, maxsize: int = 8, **kwargs) -> AsyncIterator[T]:
    """Bridge a blocking generator into an async one.

    faster-whisper's model.transcribe() returns a generator that only does work
    -- one CPU/GPU inference pass per segment -- when iterated. Iterating it on
    the event loop thread blocks every other coroutine (SSE streams, other
    ingestion jobs) for the full length of the transcription. This runs the
    iteration in a background thread and re-yields each item on the loop as it
    arrives, so the caller sees results incrementally instead of all at once
    at the end.

    maxsize bounds the queue: queue.Queue.put() blocks the producer THREAD (not
    the event loop) once full, so a slow consumer creates real backpressure
    instead of the whole transcript piling up in memory.
    """
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    loop = asyncio.get_running_loop()

    def producer() -> None:
        try:
            for item in fn(*args, **kwargs):
                q.put(item)
        except Exception as exc:            # re-raised on the consumer side below, not swallowed here
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = await loop.run_in_executor(None, q.get)   # blocking get, off the event loop
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item