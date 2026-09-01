from __future__ import annotations

import ctypes
import hashlib
import io
import json
from pathlib import Path

import pyhmmer
from pyhmmer import easel, plan7


def render(hits: plan7.TopHits) -> str:
    digest = hashlib.sha256()
    for format_ in ("targets", "domains", "pfam"):
        stream = io.BytesIO()
        hits.write(stream, format=format_, header=True)
        digest.update(stream.getvalue())
    return digest.hexdigest()


def main() -> None:
    package = Path(pyhmmer.__file__).resolve().parent
    library_path = package.parent / "pyhmmer.libs" / "liblibhmmer.so"
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    configure = library.p7_pipeline_IntraRowPageReleaseConfigure
    configure.argtypes = (ctypes.c_uint64,)
    configure.restype = ctypes.c_int
    reset = library.p7_pipeline_IntraRowPageReleaseResetStatistics
    reset.argtypes = ()
    reset.restype = None
    statistics = library.p7_pipeline_IntraRowPageReleaseStatistics
    statistics.argtypes = (
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    )
    statistics.restype = None

    data = package / "tests" / "data"
    with plan7.HMMFile(
        data / "hmms" / "txt" / "Thioesterase.hmm"
    ) as hmm_file:
        hmm = hmm_file.read()
    with easel.SequenceFile(
        data / "seqs" / "938293.PRJEB85.HG003687.faa",
        digital=True,
        alphabet=hmm.alphabet,
    ) as sequence_file:
        sequence = sequence_file.read_block()[26]
    row = easel.DigitalSequenceBlock(hmm.alphabet, [sequence])
    options = {
        "F1": 1.0,
        "F2": 1.0,
        "F3": 1.0,
        "T": -100.0,
        "domT": -100.0,
        "incT": -100.0,
        "incdomT": -100.0,
    }

    assert configure(0) == 0
    reset()
    baseline_pipeline = plan7.Pipeline(hmm.alphabet, **options)
    expected = [
        render(baseline_pipeline.search_hmm(hmm, row)) for _ in range(2)
    ]
    calls = ctypes.c_uint64()
    advised_bytes = ctypes.c_uint64()
    statistics(ctypes.byref(calls), ctypes.byref(advised_bytes))
    assert (calls.value, advised_bytes.value) == (0, 0)

    assert configure(1) == 0
    reset()
    candidate_pipeline = plan7.Pipeline(hmm.alphabet, **options)
    observed = [
        render(candidate_pipeline.search_hmm(hmm, row)) for _ in range(2)
    ]
    statistics(ctypes.byref(calls), ctypes.byref(advised_bytes))
    assert observed == expected
    assert expected[0] == expected[1]
    assert calls.value > 0
    assert advised_bytes.value > 0
    assert advised_bytes.value % 4096 == 0
    assert configure(0) == 0

    print(
        json.dumps(
            {
                "status": "PASS",
                "repetitions": len(observed),
                "output_sha256": observed[0],
                "madvise_calls": calls.value,
                "madvise_bytes": advised_bytes.value,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
