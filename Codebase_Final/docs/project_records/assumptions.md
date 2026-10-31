# Paper Assumptions

These are the three paper assumptions used consistently throughout the
methodology.

1. **Soft positives:** Three lightweight single-edit variants are pre-generated
   and stored for each reference image for soft-positive training pairs.
2. **Pair priority:** Pairs are processed in shuffled mini-batches within each
   epoch. At the end of an epoch, pair distances are recorded and decayed as
   priorities, then used to select the active pair set for the following epoch.
3. **Backbone:** The model uses the ConvNeXt-Tiny configuration: stage depths
   `[3, 3, 9, 3]` and channel widths `[96, 192, 384, 768]`.
