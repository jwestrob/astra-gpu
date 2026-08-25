"""Focused H200 oracle for the resident postfilter/F2 -> Forward seam."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline

from tests.audit_h200_phase1a_sparse_v3 import (
    OPTIONS,
    SELECTION,
    atomic_json,
    fixture,
    git_record,
    pipeline,
    require_environment,
    sha256_file,
)


COUNTERS = (
    "f2_compaction_run_count",
    "f2_source_count",
    "f2_selected_count",
    "f2_compiled_profile_count",
    "f2_unsupported_profile_count",
    "f2_selected_d2h_bytes",
    "forward_resident_f2_call_count",
    "forward_resident_f2_candidate_count",
    "forward_eliminated_candidate_h2d_bytes",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = git_record(root)
    if source["dirty"]:
        raise RuntimeError("the H200 oracle requires a clean source revision")
    environment = require_environment()
    hmms, targets = fixture()
    alphabet = hmms[0].alphabet
    before = _native._postfilter_forward_residency_statistics()

    with tempfile.TemporaryDirectory(prefix="phase3-resident-f2-") as temporary:
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
                telemetry=True,
                sparse_journal_v3=True,
            )
            exact_rows = []
            for row, profile_ordinal in enumerate(SELECTION):
                actual = candidates.search(row, pipeline(alphabet))
                expected = pipeline(alphabet).search_hmm(
                    hmms[profile_ordinal], targets
                )
                actual_hash = (
                    _pipeline._semantic_tophits_fingerprint_bound(actual).hex()
                )
                expected_hash = (
                    _pipeline._semantic_tophits_fingerprint_bound(expected).hex()
                )
                if actual_hash != expected_hash:
                    raise AssertionError(
                        f"resident F2 output differs at profile {row}"
                    )
                exact_rows.append(
                    {
                        "row": row,
                        "profile_ordinal": profile_ordinal,
                        "tophits_sha256": actual_hash,
                    }
                )

    after = _native._postfilter_forward_residency_statistics()
    delta = {key: after[key] - before[key] for key in COUNTERS}
    selected = delta["f2_selected_count"]
    if (
        delta["f2_compaction_run_count"] != 1
        or delta["f2_source_count"] <= selected
        or selected == 0
        or delta["f2_compiled_profile_count"] != len(SELECTION)
        or delta["f2_unsupported_profile_count"] != 0
        or delta["f2_selected_d2h_bytes"] != selected * 4
        or delta["forward_resident_f2_call_count"] != 1
        or delta["forward_resident_f2_candidate_count"] != selected
        or delta["forward_eliminated_candidate_h2d_bytes"] != selected * 12
    ):
        raise AssertionError(f"resident F2 counters do not reconcile: {delta}")

    record = {
        "schema_version": 1,
        "verdict": "PASS",
        "source": {
            **source,
            "native_sha256": sha256_file(Path(_native.__file__).resolve()),
            "pipeline_sha256": sha256_file(Path(_pipeline.__file__).resolve()),
        },
        "environment": environment,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "counters": delta,
        "exact_rows": exact_rows,
    }
    atomic_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
