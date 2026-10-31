# Q-selection archive status

The paper-authoritative evaluation protocol is **19 query conditions**:

- 1 original (unmodified) query
- 18 distinct transformation attacks

The horizontal flip is represented only once, as `flip_h`.  Do not add a
separate `flip` row: it is an alias of the same transformation and would
incorrectly create a 20-row average.

Use [paper_19_condition_results.md](paper_19_condition_results.md) for the
figures reported in `Reports/paper1_ieee_format.tex`.

The per-dataset `*_report_variation_selected.json` files preserve an earlier
20-row export. They must not be used to populate the paper's tables,
averages, bands, or conclusions.
