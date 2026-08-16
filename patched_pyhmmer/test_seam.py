from __future__ import annotations

import io
import importlib.metadata
from pathlib import Path

import pyhmmer
from pyhmmer import easel, plan7

import seam_probe


DATA = Path(pyhmmer.__file__).parent / "tests" / "data"
COUNTERS = ("n_past_msv", "n_past_bias", "n_past_vit", "n_past_fwd")
PIPELINE_STATE = (
    "Z",
    "domZ",
    "nmodels",
    "nseqs",
    "nres",
    *COUNTERS,
)


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


def pipeline_state(hits) -> tuple[int | float, ...]:
    state = hits.__getstate__()["pipeline"]
    return tuple(state[key] for key in PIPELINE_STATE)


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
    assert pipeline_state(standard) == pipeline_state(resumed), label
    if expected_counters is not None:
        assert counters(standard) == expected_counters, label


def compare_forward(
    label: str,
    hmm,
    sequences,
    options: dict,
    omit_forward_matrix: bool = False,
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
    resumed = seam_probe.search_forward(
        plan7.Pipeline(hmm.alphabet, **options),
        hmm,
        optimized,
        sequences,
        omit_forward_matrix,
    )
    assert tables(standard) == tables(resumed), label
    assert pipeline_state(standard) == pipeline_state(resumed), label
    if expected_counters is not None:
        assert counters(standard) == expected_counters, label


def compare_simple_regions(
    label: str,
    hmm,
    sequences,
    options: dict,
    expected_routes: tuple[int, int, int],
) -> None:
    standard = plan7.Pipeline(hmm.alphabet, **options).search_hmm(
        hmm,
        sequences,
    )
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    resumed, routes = seam_probe.search_simple_regions(
        plan7.Pipeline(hmm.alphabet, **options),
        hmm,
        optimized,
        sequences,
    )
    assert routes == expected_routes, label
    assert tables(standard) == tables(resumed), label
    assert pipeline_state(standard) == pipeline_state(resumed), label


def compare_compact_domains(
    label: str,
    hmm,
    sequences,
    options: dict,
    expected_routes: tuple[int, int, int, int] | None,
    repetitions: int = 1,
) -> None:
    standard_pipeline = plan7.Pipeline(hmm.alphabet, **options)
    compact_pipeline = plan7.Pipeline(hmm.alphabet, **options)
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    for repetition in range(repetitions):
        standard = standard_pipeline.search_hmm(hmm, sequences)
        resumed, routes = seam_probe.search_compact_domains(
            compact_pipeline,
            hmm,
            optimized,
            sequences,
        )
        if expected_routes is None:
            assert sum(routes) == len(sequences), (label, routes)
            assert routes[3] > 0, (label, routes)
        else:
            assert routes == expected_routes, (label, routes)
        assert tables(standard) == tables(resumed), (label, repetition)
        assert pipeline_state(standard) == pipeline_state(resumed), (
            label,
            repetition,
        )


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

    hmm, sequences = load("Thioesterase.hmm", 1992)
    compare_forward("external Forward", hmm, sequences, {})
    compare_forward(
        "external Forward no-bias",
        hmm,
        sequences,
        {"bias_filter": False},
    )
    compare_forward(
        "F3 reject without matrix",
        hmm,
        sequences,
        {"F1": 1.0, "F2": 1.0, "F3": -1.0},
        omit_forward_matrix=True,
        expected_counters=(1, 1, 1, 0),
    )
    compare_forward(
        "F3 negative-infinity reject without matrix",
        hmm,
        sequences,
        {"F1": 1.0, "F2": 1.0, "F3": float("-inf")},
        omit_forward_matrix=True,
        expected_counters=(1, 1, 1, 0),
    )
    for label, threshold in (
        ("F3 NaN with matrix", float("nan")),
        ("F3 positive infinity with matrix", float("inf")),
    ):
        compare_forward(
            label,
            hmm,
            sequences,
            {"F1": 1.0, "F2": 1.0, "F3": threshold},
        )
        try:
            seam_probe.search_forward(
                plan7.Pipeline(
                    hmm.alphabet,
                    F1=1.0,
                    F2=1.0,
                    F3=threshold,
                ),
                hmm,
                hmm.to_profile(
                    plan7.Background(hmm.alphabet),
                    L=400,
                ).to_optimized(),
                sequences,
                True,
            )
        except RuntimeError as error:
            assert str(error) == f"HMMER status {seam_probe.HMMER_EINVAL}"
        else:
            raise AssertionError(f"{label} accepted a missing matrix")

    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    rescale_text = "".join(
        residue if index % 20 < 8 else amino_acids[index % 20]
        for index, residue in enumerate(hmm.consensus.upper())
    )
    rescale_sequence = easel.TextSequence(
        name=b"rescale",
        sequence=rescale_text,
    ).digitize(hmm.alphabet)
    rescale_sequences = easel.DigitalSequenceBlock(
        hmm.alphabet,
        [rescale_sequence],
    )
    assert (
        seam_probe.forward_rescale_count(
            plan7.Pipeline(hmm.alphabet),
            optimized.copy(),
            rescale_sequences,
        )
        == 1
    )
    compare_forward(
        "external Forward with rescaling",
        hmm,
        rescale_sequences,
        {},
    )
    for corruption in range(6):
        status, changed = seam_probe.invalid_forward_status(
            plan7.Pipeline(hmm.alphabet),
            hmm,
            optimized.copy(),
            sequences,
            corruption,
        )
        assert status == seam_probe.HMMER_EINVAL, corruption
        assert not changed, corruption

    hmm, all_sequences = load("Thioesterase.hmm")
    route_sequences = easel.DigitalSequenceBlock(
        hmm.alphabet,
        [all_sequences[index] for index in (0, 13, 31, 67)],
    )
    all_pass = {"F1": 1.0, "F2": 1.0, "F3": 1.0}
    compare_simple_regions(
        "external simple regions",
        hmm,
        route_sequences,
        all_pass,
        (1, 1, 2),
    )
    compare_simple_regions(
        "external simple regions current options",
        hmm,
        route_sequences,
        {
            **all_pass,
            "bias_filter": False,
            "null2": False,
            "T": -100.0,
            "domT": -100.0,
            "incT": -100.0,
            "incdomT": -100.0,
        },
        (1, 1, 2),
    )
    compare_compact_domains(
        "external compact domains",
        hmm,
        route_sequences,
        all_pass,
        (1, 1, 0, 2),
    )
    compare_compact_domains(
        "external compact domains current options",
        hmm,
        route_sequences,
        {
            **all_pass,
            "bias_filter": False,
            "null2": False,
            "T": -100.0,
            "domT": -100.0,
            "incT": -100.0,
            "incdomT": -100.0,
        },
        (1, 1, 0, 2),
    )
    compare_compact_domains(
        "external compact domains fixed search spaces",
        hmm,
        route_sequences,
        {**all_pass, "Z": 123.0, "domZ": 17.0},
        (1, 1, 0, 2),
    )
    compact_fixture = easel.DigitalSequenceBlock(
        hmm.alphabet,
        [all_sequences[index] for index in range(80)],
    )
    compare_compact_domains(
        "external compact domains broad fixture and state reuse",
        hmm,
        compact_fixture,
        all_pass,
        None,
        repetitions=2,
    )
    multidomain_fixture = easel.DigitalSequenceBlock(
        hmm.alphabet,
        [all_sequences[26]],
    )
    multidomain_options = {
        **all_pass,
        "T": -100.0,
        "domT": -100.0,
        "incT": -100.0,
        "incdomT": -100.0,
    }
    multidomain_standard = plan7.Pipeline(
        hmm.alphabet,
        **multidomain_options,
    ).search_hmm(hmm, multidomain_fixture)
    assert [len(hit.domains) for hit in multidomain_standard] == [2]
    compare_compact_domains(
        "external compact multi-domain target",
        hmm,
        multidomain_fixture,
        multidomain_options,
        (0, 0, 0, 1),
    )
    invalid_sequence = easel.DigitalSequenceBlock(
        hmm.alphabet,
        [all_sequences[13]],
    )
    for corruption in (*range(27), *range(28, 37)):
        optimized = hmm.to_profile(
            plan7.Background(hmm.alphabet),
            L=400,
        ).to_optimized()
        status, changed = seam_probe.invalid_simple_regions_status(
            plan7.Pipeline(hmm.alphabet, **all_pass),
            hmm,
            optimized,
            invalid_sequence,
            corruption,
        )
        assert status == seam_probe.HMMER_EINVAL, corruption
        assert not changed, corruption
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    status, changed = seam_probe.invalid_simple_regions_status(
        plan7.Pipeline(hmm.alphabet, **all_pass),
        hmm,
        optimized,
        invalid_sequence,
        27,
    )
    assert status == seam_probe.GPU_VITERBI_OK
    assert changed

    for corruption in range(41):
        optimized = hmm.to_profile(
            plan7.Background(hmm.alphabet),
            L=400,
        ).to_optimized()
        status, changed = seam_probe.invalid_compact_domains_status(
            plan7.Pipeline(hmm.alphabet, **all_pass),
            hmm,
            optimized,
            invalid_sequence,
            corruption,
        )
        assert status == seam_probe.HMMER_EINVAL, corruption
        assert not changed, corruption
    optimized = hmm.to_profile(
        plan7.Background(hmm.alphabet),
        L=400,
    ).to_optimized()
    status, changed = seam_probe.invalid_compact_domains_status(
        plan7.Pipeline(hmm.alphabet, **all_pass),
        hmm,
        optimized,
        invalid_sequence,
        41,
    )
    assert status == seam_probe.GPU_VITERBI_OK
    assert changed

    cutoff_hmm, cutoff_sequences = load("LuxC.hmm")
    cutoff_fixture = easel.DigitalSequenceBlock(
        cutoff_hmm.alphabet,
        [cutoff_sequences[index] for index in range(16)],
    )
    compare_compact_domains(
        "external compact domains model cutoffs",
        cutoff_hmm,
        cutoff_fixture,
        {**all_pass, "bit_cutoffs": "gathering"},
        None,
    )

    check_one_library()
    print("patched PyHMMER seam parity: full searches and guarded seam cases")


if __name__ == "__main__":
    main()
