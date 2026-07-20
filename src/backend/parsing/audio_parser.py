# parsing/audio_parser.py
"""Audio -> Chunk, streamed as segments transcribe rather than returned after the
whole file finishes.

Bypasses RawElement / group_by_title on purpose: Whisper's natural chunking unit is
time (segment start/end), not headings, and audio has no Title/Table/Image elements
to flush on. Packing straight into Chunk also means a caller can start
embedding/indexing minute 0 of a 60-minute file before minute 58 has transcribed.
"""
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from faster_whisper import BatchedInferencePipeline, WhisperModel

from pipelines.text_pipeline import Chunk, ChunkMetadata
from core.config import WHISPER_BATCH, WHISPER_COMPUTE, WHISPER_DEVICE, WHISPER_MODEL
from utils.async_utils import iter_in_thread

logger = logging.getLogger(__name__)

# Same code path on Mac dev and the CUDA worker pool -- only WHISPER_* env vars differ.
# Mac (no Apple GPU backend in CTranslate2, CPU only):
#   WHISPER_DEVICE=cpu WHISPER_COMPUTE=int8 WHISPER_MODEL=small WHISPER_BATCH=8
# Prod (cuda worker, queue=video): defaults in core/config.py.
MAX_CHARS = 1500   # pack cap -- keeps each streamed chunk embeddable without a second split pass

_model: BatchedInferencePipeline | None = None


def _get_model() -> BatchedInferencePipeline:
    # Lazy singleton: importing this module must not load a multi-GB model when a
    # process only needs text/md/html support.
    global _model
    if _model is None:
        logger.info("Loading whisper model=%s device=%s compute=%s", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
        base = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        _model = BatchedInferencePipeline(model=base)
    return _model


def _pack(buf: list, idx: int, filename: str) -> Chunk:
    return Chunk(
        kind="text",
        idx=idx,
        text=" ".join(s.text.strip() for s in buf),
        metadata=ChunkMetadata(
            filename=filename,
            element_ids=[f"seg_{s.id}" for s in buf],
            start_sec=buf[0].start,   # real timestamps -- this is why audio never goes through
            end_sec=buf[-1].end,      # split_oversized, which would clone one timestamp pair across every piece
        ),
    )


async def transcribe_stream(
    path: str,
    filename: str | None = None,
    max_chars: int = MAX_CHARS,
) -> AsyncIterator[Chunk]:
    """Yields Chunks as soon as enough segments accumulate -- not after the full file transcribes."""
    name = filename or Path(path).name
    model = _get_model()

    def _run():
        segments, info = model.transcribe(path, batch_size=WHISPER_BATCH)
        logger.info("Transcribing %r language=%s duration=%.1fs", path, info.language, info.duration)
        return segments   # generator; VAD runs by default for batched transcription

    buf: list = []
    idx = 0
    async for seg in iter_in_thread(_run):
        buf.append(seg)
        if sum(len(s.text) for s in buf) >= max_chars:
            yield _pack(buf, idx, name)
            idx += 1
            buf = []
    if buf:
        yield _pack(buf, idx, name)