# JTA Person-Detection Evaluation

## Requirements

This additive protocol evaluates unordered per-frame 3D human predictions on
raw JTA people. It owns the method-neutral typed input, JTA GT/K/bbox/visibility
wiring, OKS association, completeness partition, 3D metric reduction,
sufficient statistics, and result schema. A method adapter supplies only
normalized predictions and an opaque producer/profile identity; this module
does not import, parse, or name a method-private result format. Emitting the
typed prediction is an evaluation-readiness claim: the producer must already
have source-grounded its parameter, coordinate, scale, and joint semantics or
report not-evaluation-ready instead.

The initial consumer is `hj-tpa-crowd3d`, but the contract has no Crowd3D
field. It complements the existing track-aligned JTA evaluator: predictions
have no GT identity before association, so matching is part of this protocol.
One run binds an exact ordered `(scene, frame)` population manifest and an
opaque producer/profile identity into the result digest. Publication succeeds
only after every declared frame key is consumed exactly once. An absent frame
payload is an incomplete run and fails publication; an unmatched GT person in
a consumed frame is a retained detection miss. Missing GT people and excess
predictions never shrink the evaluated view.

## Mathematical Architecture

### Normalized frame input

The evaluation-owned JTA GT frame contains:

```text
scene_id                 nonempty string
frame_id                 nonnegative integer
gt_source_ids            int64 [G], unique and ascending
gt_xy                    float64 [G,12,2], pixels
gt_visible               bool [G,12]
gt_xyz_camera            float64 [G,12,3], metres
gt_bbox_xyxy             float64 [G,4], native unpadded source boxes
camera_K                 float64 [3,3]
```

The independent method prediction frame contains the same scene/frame key,
opaque `prediction_source_sha256` and `prediction_profile_sha256` identities,
`prediction_row_ids int64 [P]`, and `pred_xyz_camera float64 [P,12,3]` in
metres. Every typed frame has a semantic digest over identity and all arrays.
The reducer joins the two typed contracts by exact frame identity and binds the
ordered frame digests into the final result. Method code never constructs the
JTA GT contract.

The endpoint order is
`L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle`.
For SMPL-54 predictions, its exact index vector is
`[16,17,18,19,20,21,1,2,4,5,7,8]`; hips are the kinematic
`L_Hip_SMPL/R_Hip_SMPL` slots 1/2. This matches the existing JTA eval metadata
and is independent of any method's training-target hip convention.

The JTA GT constructor admits one person-frame only when all 22 raw 2D and all
22 raw camera-space 3D joints are finite, the unpadded min/max envelope of all
22 raw 2D joints has positive area, and that envelope intersects the
`1920 x 1080` raster. The exact JTA-22 endpoint vector is
`[8,4,9,5,10,6,19,16,20,17,21,18]`. `gt_bbox_xyxy` is that unmodified
22-joint envelope, not the dataset-standard 15%-expanded generic bbox.
`gt_visible[g,j]` is true exactly when both source occlusion flags are zero and
`0 <= x < 1920, 0 <= y < 1080`. The only admitted intrinsic is
`[[1158,0,960],[0,1158,540],[0,0,1]]`, and projection is
`uv = (K @ X)[:2] / X[2]`.

The public evaluation-owned constructor is
`make_jta_person_detection_gt_frame(scene_id, frame_id, source_person_ids,
joints_2d_22, joints_3d_camera_22, occluded_22, self_occluded_22)`. The arrays
have leading person dimension and exact tails `[22,2]`, `[22,3]`, `[22]`, and
`[22]`; the constructor performs admission and derives K, endpoint arrays,
visibility, and bboxes. The TPA coordinator calls this public constructor after
reading raw JTA arrays; it does not reproduce the construction.

Inputs are finite and exact-shaped. A GT bbox must have positive area. A
prediction with any endpoint depth `z <= 0` is projection-invalid: it receives
no association edge and remains an unmatched prediction. Empty GT and
prediction arrays are valid boundary cases.

### OKS and association

Project valid predictions through `camera_K`. Pairwise OKS uses only true
entries of `gt_visible`, native GT bbox area, and sigmas:

```text
[0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
```

A GT with zero visible endpoints has no admissible edge. Edges with
`OKS < 0.50` are inadmissible before assignment. The matching objective is
lexicographic:

1. maximize accepted cardinality;
2. maximize total accepted quantized OKS;
3. over ascending GT source IDs, lexicographically minimize assigned
   prediction row, with unmatched ordered after all real rows.

Before optimization, each admissible binary64 OKS is quantized as
`q = rint(OKS * 10^12)` using round-to-nearest ties-to-even. The first two
objectives are therefore exact maximum cardinality and maximum integer sum of
`q`. This second objective is deliberately the sum of per-edge quantized
values, not the raw binary64 OKS sum; per-edge quantization can change an
aggregate ordering near a rounding boundary and that behavior is part of the
profile.
Reported OKS retains the original binary64 value. The first two objectives use
`scipy.optimize.linear_sum_assignment` on an augmented threshold graph. The
third objective uses explicit optimum-preserving refinement against exact
integer sums, not incidental solver tie behavior. It examines only candidates
earlier than the current assignment; the hard upper bound is `G * (P + 1)`
additional solver calls, while an already lex-minimal unique solution needs
none. In particular the matrix
`[[0.90,0.51],[0.50,0.49]]` must yield two accepted cross pairs, not one
diagonal pair.

### Metrics and empty semantics

Report matched mean OKS; all-GT mean OKS with unmatched GT contributing zero;
recall; unmatched GT, unmatched prediction, and projection-invalid prediction
counts; and matched-person occurrence-macro absolute MPJPE, pelvis-aligned
MPJPE, and PA-MPJPE. Each 3D occurrence uses all twelve endpoints and reports
millimetres after multiplying source metres by 1000.

Pelvis alignment subtracts prediction and GT's own midpoint of endpoint slots
6/7 independently. PA fits prediction to GT per occurrence with positive
isotropic scale and reflection disabled. A PA occurrence is degenerate only
when every prediction joint equals its first prediction joint, or every GT
joint equals its first GT joint, under direct `float64` equality. This exact
zero-spread predicate is checked before calling the geometry leaf. Degenerate fits
are counted and excluded only from the PA denominator; they do not erase
absolute or pelvis-aligned errors. Every exception from
`fit_similarity_registration`, including non-positive correlation, SVD
failure, overflow, or non-finite fitted state, remains a hard failure. If all
accepted occurrences are PA-degenerate, PA mean is `null` and both PA person
and joint denominators are zero.

An empty expected-frame manifest is a constructor error. When there is no
accepted match, matched-only means are JSON `null` with
denominator zero. When `G == 0`, all-GT OKS and recall are also `null`; when
`G > 0`, both retain denominator `G` even if no prediction matches. The result
contains explicit person and joint denominators and exact input population
identity, and validates that the matching and PA partitions are complete. The decoder
also verifies the serialized derived metrics and denominators against the
sufficient statistics rather than treating them as unchecked presentation.

## Code Architecture

- `src/hjlib_evaluation/jta_person_detection_data.py` owns immutable typed JTA
  GT and method-prediction frame inputs, raw-JTA-to-GT construction, immutable
  result/summary values, semantic digests, and canonical JSON codecs.
- `src/hjlib_evaluation/jta_person_detection_protocol.py` owns projection,
  threshold-aware deterministic association, per-frame metric computation, and
  sufficient-stat accumulation.
- The protocol reuses `scipy.optimize.linear_sum_assignment`,
  `compute_keypoint_oks_matrix`,
  `compute_joint_position_errors`, and the reviewed `hjlib-geometry`
  similarity fit/apply. It does not duplicate those mathematical leaves.
- `JTA_Person_Detection_Reducer(expected_frame_keys,
  prediction_source_sha256, prediction_profile_sha256)` is stateful only
  across that explicit binding. `add_frame(gt_frame, prediction_frame)` rejects
  duplicate/out-of-order/undeclared keys and either prediction-identity
  mismatch without mutating state on failure.
  `finalize()` refuses a prefix, returns one immutable result, and seals the
  reducer; any later `add_frame` or `finalize` fails. Library finalization does
  no file I/O. Per-frame validation and association are stateless public
  functions; canonical JSON encoding is a separate pure function.
- JSON serialization is canonical, carries schema/version and a semantic
  SHA-256, and represents undefined means as `null`, never NaN. The protocol
  keeps no method-native arrays or images.
- The public surface is re-exported by `hjlib_evaluation.__init__`. SciPy is an
  explicit runtime dependency. No dataset local setting or GT provider is
  imported by the leaf module; the TPA coordinator calls the explicit raw-array
  GT constructor and its own method-prediction adapter separately.

## Smoke-Test Standard

Data-free smoke tests cover exact raw-JTA population/bbox/visibility/K
construction and shape/dtype/index validation; perfect,
permuted, duplicate, missed, empty, zero-visible, and projection-invalid
inputs; the threshold-graph counterexample; rectangular all-equal tie cases;
native bbox-area and sigma/order pins; exact SMPL-54 names/indices; empty-run
rejection; quantization-boundary and near-but-unequal total-OKS cases;
metre-to-millimetre conversion; independent pelvis alignment;
reflection-disabled PA; degenerate PA; population ordering and duplicate
rejection; zero-denominator JSON; canonical round-trip and tamper rejection.
Reducer state tests cover incomplete-prefix and undeclared-key rejection,
validation failure without mutation, exactly-once finalization, sealed-state
write/finalize rejection, and an unequal multi-frame oracle that distinguishes
occurrence-macro sufficient-stat reduction from frame-macro averaging while
checking every person and joint denominator.
A synthetic adversarial all-tie/duplicate `G=P=128` gate must finish within
2.0 CPU seconds and at most 129 total assignment-solver calls, in addition to
the general hard bound. Strict pyright and the repository smoke suite are
mandatory.

## Migration Plan

1. Review Requirements, Mathematical Architecture, Code Architecture, and the
   Smoke-Test Standard before implementation.
2. Add the typed leaf/reducer and data-free tests without changing existing
   JTA track-aligned evaluation behavior.
3. Re-export the additive API, document usage, run repository gates, and land
   the leaf before updating the consuming TPA dependency pin.

## Modification History

- 2026-08-23: Initial design split method-private prediction interpretation
  from evaluation-owned association and reduction. It freezes SMPL-24 hip
  semantics, threshold-aware maximum-cardinality OKS association, deterministic
  ties, unmatched populations, and metric/empty-result behavior for the JTA
  five-epoch Crowd3D comparison.
- 2026-08-23: Layer reviews froze evaluation-owned raw-JTA GT construction,
  kinematic hip indices, quantized threshold-aware matching, typed producer
  provenance, exact reducer state, PA-degeneracy and empty semantics, module
  residence, SciPy reuse, and executable performance/state smoke gates.
- 2026-08-28: Corrected the implementation to match the frozen PA contract:
  only direct equality of every joint to the person's first joint is treated
  as exact zero spread and denominator-excluded as degenerate; every
  geometry-fit exception now propagates as a hard evaluation failure.
