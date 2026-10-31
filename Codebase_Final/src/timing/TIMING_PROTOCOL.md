# Phase-3 Timing Protocol

This is the only timing procedure for final direct-vs-HNSW results.

## Invariants

1. Use the same bank, embedding dimension, threshold, current 19-attack
   manifest, and query ordering for direct and graph decisions.
2. Perform direct decisions for every Q size before the graph sweep. Those
   direct decisions are the immutable correctness oracle.
3. Measure one production-style query per kernel call. Do not time a large
   batch and divide it by Q.
4. Use one fixed stratified timing panel per Q size. The current panel samples
   five known and five unknown queries from each of 19 attacks: 190 queries/Q.
5. Every graph candidate must be checked against direct decisions across the
   entire Q workload, not merely on the timing panel.
6. Time graph candidates serially on the isolated timing lane. Cached inputs
   may be reused, but no candidate's measured latency may be reused for another.

## Direct Baseline

For each query in the fixed panel:

1. Execute one untimed priming call.
2. Execute one warmup call.
3. Time nine one-query direct calls.
4. Remove the fastest and slowest value and store the median of the remaining
   seven values.

The stored per-query direct medians are reused as the denominator for every
candidate at the same Q. Direct timing is measured separately for each Q size;
there is no q100-only denominator.

## Candidate Screening

For each `(C, overlap, degree, entries, ef)` candidate:

1. Reuse only immutable inputs: embeddings, direct truth, timing panel,
   deterministic centroid layout, overlap posting lists, BVH, and the HNSW
   graph for the same `(C, degree, efConstruction)`.
2. Time every timing-panel query after one warmup using three one-query graph
   measurements and a trimmed median.
3. Run the complete Q workload at every configured Q size and require 100%
   decision agreement with direct search.
4. Persist the candidate record even when it is not selected.

## Final Reporting Timing

For each candidate chosen for final reporting:

1. Warm every timing-panel query eight times.
2. Time it 41 times, remove the fastest and slowest values, and take the
   median of the remaining measurements.
3. Report per-Q direct and HNSW min/mean/max latency, speedup, full-workload
   match count, Q-mean latency spread, and full-workload internal tail latency.

Speedup for Q is:

```text
speedup(Q) = direct_panel_mean_ms(Q) / graph_panel_mean_ms(Q)
```

`--no-selection` performs independent full-Q validation for every planned
candidate and intentionally does not publish a final timing selection. Run the
same evaluator without that flag to robustly re-time one exact winner per
centroid before selecting a reported configuration.
