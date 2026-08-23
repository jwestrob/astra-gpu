"""Focused H200 oracle for the Phase 1A sparse journal-v3 consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline
from plan7_gpu.adapter import _candidate_state, _sequence_state


EXPECTED_TARGET = "sm90_h200"
SELECTION = (3, 1, 0)
OPTIONS = {"F1": 0.99, "F2": 1.0, "F3": 1.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline(alphabet: pyhmmer.easel.Alphabet) -> pyhmmer.plan7.Pipeline:
    return pyhmmer.plan7.Pipeline(
        alphabet,
        E=10.0,
        domE=10.0,
        incE=10.0,
        incdomE=10.0,
        **OPTIONS,
    )


def fixture() -> tuple[list[pyhmmer.plan7.HMM], object]:
    data = Path(pyhmmer.__file__).parent / "tests" / "data"
    hmms = []
    for name in ("RREFam.hmm", "Thioesterase.hmm", "KR.hmm", "LuxC.hmm"):
        with pyhmmer.plan7.HMMFile(data / "hmms" / "txt" / name) as source:
            hmm = source.read()
        if hmm is None:
            raise RuntimeError(f"missing PyHMMER fixture HMM: {name}")
        hmms.append(hmm)

    sequences = []
    with pyhmmer.easel.SequenceFile(
        data / "seqs" / "938293.PRJEB85.HG003687.faa",
        digital=True,
        alphabet=hmms[0].alphabet,
    ) as source:
        for sequence in source:
            sequences.append(sequence)
            if len(sequences) == 4:
                break
    if len(sequences) != 4:
        raise RuntimeError("PyHMMER fixture FASTA has fewer than four targets")

    consensus = hmms[1].consensus
    if isinstance(consensus, bytes):
        consensus = consensus.decode()
    sequences.append(
        pyhmmer.easel.TextSequence(
            name=b"two-domain-probe",
            sequence=consensus.replace("-", "") + "X" * 100
            + consensus.replace("-", ""),
        ).digitize(hmms[0].alphabet)
    )
    return hmms, pyhmmer.easel.DigitalSequenceBlock(
        hmms[0].alphabet, sequences
    )


def git_record(root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def require_environment() -> dict[str, object]:
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != EXPECTED_TARGET:
        raise RuntimeError("this oracle requires an attested H200/sm90 runtime")
    seams = {
        "filter": _pipeline._filter_scores_seam_available(),
        "forward": _pipeline._filter_and_forward_scores_seam_available(),
        "simple_regions": _pipeline._simple_regions_seam_available(),
        "compact_domains": _pipeline._compact_domains_seam_available(),
    }
    if not all(seams.values()):
        raise RuntimeError(f"required private continuation seam is absent: {seams}")
    return {"provenance": provenance, "seams": seams}


def audit_fixture() -> dict[str, object]:
    hmms, targets = fixture()
    alphabet = hmms[0].alphabet
    with tempfile.TemporaryDirectory(prefix="phase1a-h200-v3-") as temporary:
        base = Path(temporary) / "models"
        pyhmmer.hmmer.hmmpress(hmms, base)
        pairs = load_pressed_profiles(base)
        with (
            ProfileSession(pairs, pack_workers=1) as session,
            session.select(SELECTION) as selection,
            SequenceBatch(targets) as batch,
        ):
            candidates = batch._postfilter_forward_selection(
                selection,
                **OPTIONS,
                bias_filter=True,
                pipeline=pipeline(alphabet),
            )
            sealed = _candidate_state(candidates).sealed_postfilter
            if sealed is None:
                raise AssertionError("fused CUDA generation did not produce a seal")
            generation = _pipeline._sealed_continuation_statistics_bound(sealed)
            workspace = dict(_sequence_state(batch).native.workspace_statistics)

        packet = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            packet, sealed, include_details=True
        )
        audit = _pipeline._audit_continuation_journal_v3_bound(
            packet, sealed, pipeline(alphabet), pipeline(alphabet)
        )

    required_routes = ("forward_scores", "simple_regions", "compact_domains")
    missing_routes = [
        route for route in required_routes
        if summary["exception_routes"][route] == 0
    ]
    rows = audit["rows"]
    accepted = sum(row["dense"]["compact"]["accepted_count"] for row in rows)
    route_failures = [
        row["profile_index"]
        for row in rows
        if not row["route_reconciliation"]["equal"]
    ]
    semantic_failures = [
        row["profile_index"]
        for row in rows
        if not row["semantic"]["pipeline"]["equal"]
        or not row["semantic"]["tophits"]["equal"]
    ]
    if missing_routes:
        raise AssertionError(f"fixture missed required v3 routes: {missing_routes}")
    if accepted == 0:
        raise AssertionError("fixture did not exercise compact acceptance")
    if route_failures or semantic_failures or audit["equal"] is not True:
        raise AssertionError(
            "dense-v2/sparse-v3 mismatch: "
            f"route={route_failures}, semantic={semantic_failures}"
        )
    if workspace["forward_run_count"] <= 0 or workspace["postfilter_run_count"] <= 0:
        raise AssertionError("audit did not execute real CUDA Forward/postfilter work")

    return {
        "targets": len(targets),
        "profiles": len(SELECTION),
        "generation": generation,
        "workspace": workspace,
        "v3_summary": {
            key: summary[key]
            for key in (
                "source_kind",
                "dense_postfilter_count",
                "dense_forward_count",
                "dense_domain_count",
                "certificate_count",
                "exception_count",
                "stage_counts",
                "exception_routes",
                "payload_counts",
                "packet_bytes",
                "source_v2_bytes",
            )
        },
        "dual": {
            "equal": audit["equal"],
            "packet_bytes": audit["packet_bytes"],
            "prestate_sha256": audit["prestate_sha256"],
            "compact_accepted": accepted,
            "rows": audit["rows"],
        },
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = git_record(root)
    if source["dirty"]:
        raise RuntimeError("the H200 oracle requires a clean source revision")
    environment = require_environment()
    started = time.monotonic()
    result = audit_fixture()
    record = {
        "schema_version": 1,
        "verdict": "PASS",
        "scope": "authenticated fused CUDA v2 versus sparse journal v3",
        "source": {
            **source,
            "native_extension": {
                "path": str(Path(_native.__file__).resolve()),
                "sha256": sha256_file(Path(_native.__file__).resolve()),
            },
            "pipeline_extension": {
                "path": str(Path(_pipeline.__file__).resolve()),
                "sha256": sha256_file(Path(_pipeline.__file__).resolve()),
            },
        },
        "host": {
            "hostname": platform.node(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        },
        "environment": environment,
        "result": result,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
