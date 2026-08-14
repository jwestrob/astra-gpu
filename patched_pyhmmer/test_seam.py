from __future__ import annotations

import io
import importlib.metadata
from pathlib import Path

import pyhmmer
from pyhmmer import easel, plan7

import seam_probe


DATA = Path(pyhmmer.__file__).parent / "tests" / "data"
COUNTERS = ("n_past_msv", "n_past_bias", "n_past_vit", "n_past_fwd")


def load(hmm_name: str, index: int | None = None):
    with plan7.HMMFile(DATA / "hmms" / "txt" / hmm_name) as hmm_file:
        hmm = hmm_file.read()
    with easel.SequenceFile(
        DATA / "seqs" / "938293.PRJEB85.HG003687.faa",
        digital=True,
        alphabet=hmm.alphabet,
    ) as sequence_file:
        sequences = sequence_file.read_block()
    if index is not None:
        sequences = easel.DigitalSequenceBlock(
            hmm.alphabet,
            [sequences[index]],
        )
    return hmm, sequences


def tables(hits) -> tuple[bytes, ...]:
    output = []
    for format_ in ("targets", "domains", "pfam"):
        stream = io.BytesIO()
        hits.write(stream, format=format_, header=True)
        output.append(stream.getvalue())
    return tuple(output)


def counters(hits) -> tuple[int, ...]:
    state = hits.__getstate__()["pipeline"]
    return tuple(state[key] for key in COUNTERS)


def compare(
    label: str,
    hmm,
    sequences,
    options: dict,
    gpu_status: int = seam_probe.GPU_VITERBI_OK,
    poison_filtersc: bool = False,
    expected_counters: tuple[int, ...] | None = None,
) -> None:
    standard = plan7.Pipeline(hmm.alphabet, **options).search_hmm(
        hmm,
        sequences,
    )
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    resumed = seam_probe.search(
        plan7.Pipeline(hmm.alphabet, **options),
        hmm,
        optimized,
        sequences,
        gpu_status,
        poison_filtersc,
    )
    assert tables(standard) == tables(resumed), label
    assert counters(standard) == counters(resumed), label
    if expected_counters is not None:
        assert counters(standard) == expected_counters, label


def check_one_library() -> None:
    mapped = Path("/proc/self/maps").read_text().splitlines()
    for library in ("liblibhmmer.so", "liblibeasel.so"):
        paths = {
            str(Path(line.split()[-1]).resolve())
            for line in mapped
            if line.endswith("/" + library)
        }
        assert len(paths) == 1, (library, paths)


def main() -> None:
    assert pyhmmer.__version__ == "0.12.0"
    assert importlib.metadata.version("pyhmmer") == "0.12.0+plan7gpu.0"

    hmm, sequences = load("Thioesterase.hmm")
    for label, options in (
        ("default", {}),
        ("no-bias", {"bias_filter": False}),
        ("all-pass", {"F1": 1.0, "F2": 1.0, "F3": 1.0}),
        (
            "score-thresholds",
            {"T": -100.0, "domT": -100.0, "incT": -100.0, "incdomT": -100.0},
        ),
    ):
        compare(label, hmm, sequences, options)

    hmm, sequences = load("Thioesterase.hmm", 1992)
    standard = plan7.Pipeline(hmm.alphabet).search_hmm(hmm, sequences)
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    from_msv = seam_probe.search_from_msv(
        plan7.Pipeline(hmm.alphabet),
        hmm,
        optimized,
        sequences,
    )
    assert tables(standard) == tables(from_msv), "post-MSV"
    assert counters(standard) == counters(from_msv), "post-MSV"

    hmm, sequences = load("LuxC.hmm", 6)
    compare("bias reject", hmm, sequences, {"F1": 0.2}, expected_counters=(1, 0, 0, 0))

    hmm, sequences = load("Thioesterase.hmm", 13)
    compare("F2 reject", hmm, sequences, {}, expected_counters=(1, 1, 0, 0))

    hmm, sequences = load("Thioesterase.hmm", 67)
    compare(
        "Viterbi elided",
        hmm,
        sequences,
        {"F1": 1.0, "F2": 1.0},
        seam_probe.GPU_VITERBI_NOT_RUN,
        expected_counters=(1, 1, 1, 0),
    )

    hmm, sequences = load("Thioesterase.hmm", 1992)
    for label, status in (
        ("ERANGE fallback", seam_probe.GPU_VITERBI_ERANGE),
        ("ENORESULT fallback", seam_probe.GPU_VITERBI_ENORESULT),
    ):
        compare(label, hmm, sequences, {}, status, expected_counters=(1, 1, 1, 1))

    hmm, sequences = load("Thioesterase.hmm", 67)
    compare(
        "no-bias uses nullsc",
        hmm,
        sequences,
        {"bias_filter": False, "F1": 1.0, "F2": 1.0, "F3": 0.01},
        seam_probe.GPU_VITERBI_NOT_RUN,
        poison_filtersc=True,
        expected_counters=(1, 1, 1, 1),
    )

    check_one_library()
    print("patched PyHMMER seam parity: 4 full searches + 7 seam cases")


if __name__ == "__main__":
    main()
