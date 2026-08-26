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

Decision: retain as an isolated opt-in pending the paired full H200 oracle.
