# Request-scoped continuation workers

Hypothesis: rebuilding a `ThreadPoolExecutor` and one HMMER `Pipeline` per
worker for every profile chunk adds avoidable control-plane work. The private
opt-in keeps one worker pool per immutable Pipeline-option group for the whole
Astra request. Public `plan7_gpu.astra_search.hmmsearch` behavior and signature
are unchanged.

Local exact first-1000 A/B, one warm profile session, 63 continuation workers:

- output SHA-256: `5b4894189d5a0ec0d0c6503e1524b1d2f8018db8d2b2961ea09860b9b7d0ee64`
- retained path: request 12.121792 s; continuation 4.951478 s
- pooled path: request 11.877031 s; continuation 4.589079 s
- change: request -2.02%; continuation -7.32%
- pooled calls: 14; Pipelines constructed: 63 total

Full H200 job `1185208` reproduced the exact retained output (SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, 383,235 lines). Request wall was 400.888634 s,
generation 335.637724 s, continuation/output 337.703163 s, measured overlap
280.580475 s, and pipeline wall 392.908699 s. This was 2.168434 s (0.54%)
faster than the retained 403.057068 s run, below the threshold for a runtime
claim without repeats. Peak RSS fell from 14,345,372 KiB to 6,865,076 KiB, a
52.14% reduction, while constructing exactly 63 Pipelines for all 83 chunks.

The same-allocation paired full repeats were also exact. The unpooled arms
took 399.239473 s and 397.853595 s; pooled arms took 397.331963 s and
398.103527 s. Their means were 398.546534 s and 397.717745 s respectively, a
0.208% pooled reduction that is below the runtime noise threshold. Mean peak
RSS, however, fell from 14,576,952 KiB to 6,864,202 KiB, a reproducible 52.91%
reduction.

The decisive combination gate was full H200 job `1185307`, which paired the
pool with exact profile sharding. It reproduced the full output oracle in
342.819173 s with 7,420,788 KiB peak RSS. Relative to sharding alone, wall was
effectively flat (-0.134%) while peak RSS fell 57.94%.

Decision: retain the request-scoped pool. It is a measured memory-efficiency
win, not a standalone runtime claim, and is the retained companion to profile
sharding.
