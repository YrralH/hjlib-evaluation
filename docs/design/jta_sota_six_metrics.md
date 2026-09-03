# JTA fitted-SMPL six-metric reducer

## Contract

`jta12_fitted_all_valid_v1` evaluates paired fitted-SMPL occurrences on the
twelve JTA limb endpoints. Prediction and GT occurrence IDs must already be
paired and identical; this reducer performs no detection association and does
not drop invalid rows. World arrays may use SMPL24 or SMPL54 layout and metres
or millimetres, but prediction and GT must share one declared coordinate frame.

The reducer reports occurrence-weighted root error, MPJPE, translation-aligned
T-MPJPE, rigid-aligned RT-MPJPE, similarity-aligned PA-MPJPE, and paired 2D OKS.
The root is the midpoint of SMPL hips 1 and 2. MPJPE-family denominators count
all twelve endpoints for every occurrence; root error and OKS use one value per
occurrence. Alignment fits are reflection-disabled.

## Reduction boundary

`compute_jta_sota_metric_sums(...)` returns additive sufficient statistics for
one batch. `JTA_SOTA_Metric_Sums.plus(...)` combines compatible batches, and
`finalize_jta_sota_metric_sums(...)` performs the only division. This permits
train-step logging and whole-split evaluation to share exact denominator
semantics without storing every prediction in memory.

`validate_jta_sota_occurrence_partition(...)` is a separate completeness gate:
the ordered concatenation of batch IDs must equal the expected population. A
method adapter may implement equivalent metric math in its pinned native
runtime only after parity against this reducer.

## Extension boundary

A different joint set, root, validity policy, alignment, unit convention, OKS
area, or denominator is a new metric profile. Detection association belongs to
the existing JTA person-detection protocol and must not be folded into this
paired-occurrence reducer.
