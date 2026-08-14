from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyhmmer
from pyhmmer import easel, plan7


data = Path(pyhmmer.__file__).parent / "tests" / "data"
with plan7.HMMFile(data / "hmms" / "txt" / "Thioesterase.hmm") as hmm_file:
    hmm = hmm_file.read()
with easel.SequenceFile(
    data / "seqs" / "938293.PRJEB85.HG003687.faa",
    digital=True,
    alphabet=hmm.alphabet,
) as sequence_file:
    sequences = sequence_file.read_block()

cases = (
    {},
    {"bias_filter": False},
    {"F1": 1.0, "F2": 1.0, "F3": 1.0},
    {"T": -100.0, "domT": -100.0, "incT": -100.0, "incdomT": -100.0},
)
signature = []
for options in cases:
    hits = plan7.Pipeline(hmm.alphabet, **options).search_hmm(hmm, sequences)
    digest = hashlib.sha256()
    for format_ in ("targets", "domains", "pfam"):
        stream = io.BytesIO()
        hits.write(stream, format=format_, header=True)
        digest.update(stream.getvalue())
    pipeline = hits.__getstate__()["pipeline"]
    signature.append(
        (
            len(hits),
            digest.hexdigest(),
            tuple(
                pipeline[key]
                for key in (
                    "n_past_msv",
                    "n_past_bias",
                    "n_past_vit",
                    "n_past_fwd",
                )
            ),
        )
    )

print(json.dumps(signature, separators=(",", ":")))
