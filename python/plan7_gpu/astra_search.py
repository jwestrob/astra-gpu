"""Ordered candidate-aware ``hmmsearch`` for Astra's pressed-database path.

This module intentionally stops at the same boundary as
``pyhmmer.hmmsearch``: :func:`hmmsearch` accepts already-loaded pressed
profiles and an already-parsed target/candidate batch, then yields one
``TopHits`` object per query.  Astra remains responsible for parsing targets,
grouping model-specific cutoffs, and writing results.

A :class:`~plan7_gpu.adapter.SequenceBatch` is convenient when the caller
wants this function to build the candidate rows.  Passing a precomputed
:class:`~plan7_gpu.adapter.CandidateBatch` avoids filtering twice; its bound
profile identities and F1 threshold are checked before any CPU search starts.
Parallel searches use small contiguous row chunks, a two-chunks-per-worker
reorder window, and one exclusively owned, exact-base PyHMMER ``Pipeline`` per
worker thread.  The window keeps workers busy across query skew without
retaining an unbounded number of completed ``TopHits`` objects behind one slow
query.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
import operator
from threading import local
from typing import Any

import pyhmmer

from .adapter import (
    CandidateBatch,
    PressedProfilePair,
    SequenceBatch,
    _candidate_state,
)


_MAX_ROWS_PER_TASK = 8
_TASKS_PER_WORKER_WINDOW = 2


def _positive_cpus(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("cpus must be a positive integer, not bool")
    try:
        cpus = operator.index(value)
    except TypeError as error:
        raise TypeError("cpus must be a positive integer") from error
    if cpus <= 0:
        raise ValueError(f"cpus must be strictly positive, not {cpus!r}")
    return cpus


def _pressed_pairs(
    profile_pairs: Iterable[PressedProfilePair],
) -> tuple[PressedProfilePair, ...]:
    try:
        pairs = tuple(profile_pairs)
    except TypeError as error:
        raise TypeError("profile_pairs must be an iterable") from error
    for pair in pairs:
        if type(pair) is not PressedProfilePair:
            raise TypeError(
                "profile_pairs must contain only objects from load_pressed_profiles"
            )
    return pairs


def _same_bound_pairs(
    pairs: tuple[PressedProfilePair, ...], candidates: CandidateBatch
) -> bool:
    bound_pairs = _candidate_state(candidates).pairs
    return len(pairs) == len(bound_pairs) and all(
        supplied is bound for supplied, bound in zip(pairs, bound_pairs, strict=True)
    )


def _prepare_candidates(
    pairs: tuple[PressedProfilePair, ...],
    batch: SequenceBatch | CandidateBatch,
    pipeline_options: dict[str, Any],
    postfilter: bool,
) -> CandidateBatch:
    if type(batch) is SequenceBatch:
        build = batch.postfilter_batch if postfilter else batch.candidate_batch
        candidates = build(pairs, F1=pipeline_options.get("F1", 0.02))
        alphabet = batch.alphabet
    elif type(batch) is CandidateBatch:
        candidates = batch
        if not _same_bound_pairs(pairs, candidates):
            raise ValueError(
                "candidate batch is not bound to the supplied profile pairs "
                "in the same order"
            )
        alphabet = _candidate_state(candidates).targets.alphabet
        if "F1" in pipeline_options:
            try:
                requested_f1 = float(pipeline_options["F1"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("pipeline F1 must match candidate batch F1") from error
            if requested_f1 != candidates.F1:
                raise ValueError(
                    f"pipeline F1 {requested_f1!r} does not match "
                    f"candidate batch F1 {candidates.F1!r}"
                )
    else:
        raise TypeError("batch must be exactly SequenceBatch or CandidateBatch")

    # Use the exact normalized threshold stored with the candidate rows.  This
    # matters when a precomputed batch used a non-default F1 and the caller did
    # not otherwise need to repeat that option.
    pipeline_options["F1"] = candidates.F1
    if "alphabet" in pipeline_options:
        if pipeline_options["alphabet"] != alphabet:
            raise ValueError(
                "pipeline alphabet does not match the bound target alphabet"
            )
    else:
        pipeline_options["alphabet"] = alphabet
    return candidates


def _search_row(candidates: CandidateBatch, row: int, pipeline: Any) -> Any:
    try:
        hits = candidates.search(row, pipeline)
    except BaseException:
        # Match PyHMMER's worker lifecycle and leave no dirty pipeline queued
        # for another row if the failure happened after pipeline mutation.
        try:
            pipeline.clear()
        except BaseException:
            pass
        raise
    pipeline.clear()
    return hits


def _serial_hmmsearch(
    candidates: CandidateBatch,
    pipeline_options: dict[str, Any],
) -> Iterator[Any]:
    pipeline = pyhmmer.plan7.Pipeline(**pipeline_options)
    for row in range(len(candidates)):
        yield _search_row(candidates, row, pipeline)


def _threaded_hmmsearch(
    candidates: CandidateBatch,
    cpus: int,
    pipeline_options: dict[str, Any],
) -> Iterator[Any]:
    worker_count = min(cpus, len(candidates))
    worker_state = local()

    # Eight rows is enough to amortize Future creation for the many cheap
    # reject-only rows common after GPU filtering, while keeping retained
    # TopHits and query-skew latency small.  Shrink chunks for short searches
    # so there are up to two initial tasks per worker instead of idling most
    # workers behind one oversized chunk.
    chunk_size = min(
        _MAX_ROWS_PER_TASK,
        max(
            1,
            (len(candidates) + _TASKS_PER_WORKER_WINDOW * worker_count - 1)
            // (_TASKS_PER_WORKER_WINDOW * worker_count),
        ),
    )

    def search_chunk(start: int, stop: int) -> tuple[list[Any], BaseException | None]:
        pipeline = getattr(worker_state, "pipeline", None)
        if pipeline is None:
            pipeline = pyhmmer.plan7.Pipeline(**pipeline_options)
            worker_state.pipeline = pipeline

        hits = []
        for row in range(start, stop):
            try:
                hits.append(_search_row(candidates, row, pipeline))
            except BaseException as error:
                # A task must preserve row-wise failure order: successful rows
                # before the failing row are yielded before this exact error.
                return hits, error
        return hits, None

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="plan7-gpu-astra",
    )

    def submit(start: int) -> Future[Any]:
        stop = min(start + chunk_size, len(candidates))
        try:
            return executor.submit(search_chunk, start, stop)
        except BaseException as error:
            failed: Future[Any] = Future()
            failed.set_exception(error)
            return failed

    pending: deque[Future[Any]] = deque()
    next_row = 0
    window_size = min(
        len(candidates),
        chunk_size * _TASKS_PER_WORKER_WINDOW * worker_count,
    )
    try:
        while next_row < window_size:
            pending.append(submit(next_row))
            next_row += chunk_size

        while pending:
            future = pending.popleft()
            chunk_hits, error = future.result()
            # Refill the strict chunk window before handing successful results
            # to Astra, so workers keep running while Astra writes the hits.
            # Do not schedule more work after a known row failure.
            if error is None and next_row < len(candidates):
                pending.append(submit(next_row))
                next_row += chunk_size
            for hits in chunk_hits:
                yield hits
            if error is not None:
                raise error
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def hmmsearch(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    cpus: int = 1,
    postfilter: bool = False,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Yield candidate-aware HMM searches in supplied query order.

    Parameters mirror the narrow portion of ``pyhmmer.hmmsearch`` replaced in
    Astra's installed pressed-database bulk path.  ``pipeline_options`` are
    forwarded unchanged to exact base :class:`pyhmmer.plan7.Pipeline` objects,
    except that ``alphabet`` is inferred when absent and ``F1`` is normalized
    to the threshold bound to the candidate rows.

    ``cpus`` is deliberately stricter than PyHMMER's convenience function: it
    must be a positive integer, with ``1`` executing lazily in the consuming
    thread.  Values of ``0`` are rejected instead of consulting host topology,
    so Astra's explicit ``--threads`` allocation cannot be exceeded silently.
    Threaded iteration keeps at most twice the effective worker count in-flight
    chunks of at most eight rows each, bounding both the reorder buffer and
    queued work.  The returned iterator owns all pipelines and worker threads
    until it is exhausted or closed.
    """
    worker_count = _positive_cpus(cpus)
    if type(postfilter) is not bool:
        raise TypeError("postfilter must be bool")
    pairs = _pressed_pairs(profile_pairs)
    options = dict(pipeline_options)
    candidates = _prepare_candidates(pairs, batch, options, postfilter)

    if len(candidates) != len(pairs):
        # This should be guaranteed by CandidateBatch construction, but keep
        # the bridge's row/query contract explicit at its trust boundary.
        raise ValueError("candidate row count does not match profile pair count")
    if not pairs:
        return iter(())
    if worker_count == 1:
        return _serial_hmmsearch(candidates, options)
    return _threaded_hmmsearch(candidates, worker_count, options)


__all__ = ["hmmsearch"]
