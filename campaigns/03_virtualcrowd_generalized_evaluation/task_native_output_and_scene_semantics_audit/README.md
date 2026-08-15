# Task - Native Output And Scene Semantics Audit

## Purpose And Boundary

Resolve the native-artifact facts needed before selecting the corrected
VirtualCrowd protocol:

1. determine whether dense `pose`/`translation` values at
   `track_flag == False` are final reconstructed outputs, interpolated or
   optimized latent states, copied/stale values, or invalid placeholders;
2. inventory every ground/scene/camera representation carried by the supplied
   Crowd4D and DyCrowd artifacts and classify it as method-predicted,
   method-optimized, copied method input, shared scene input, or GT;
3. determine which representation, if any, provides a sequence-constant
   scene/world coordinate suitable for corrected spatial and temporal metrics;
4. state the remaining whole-sequence association contract. Association is a
   designable adapter operation and is not treated as an unsolved source-data
   blocker.

This task is read-only. It does not alter a metric formula, implement an
adapter, modify the supplied artifacts, run inference, or run corrected
evaluation.

## Owner And Residence Boundary

- `hjlib-evaluation` owns this campaign decision and the consequences for the
  method-neutral evaluation contract.
- `hj-tpa-crowd4d` owns any Crowd4D/DyCrowd-native parsing or later probe
  implementation. Native keys and layouts must not become public
  `hjlib-evaluation` contracts.
- The supplied package remains an immutable external input. Existing T2 and
  GT-MOT evidence remain frozen historical results.

## State

- State: complete
- Result: both methods carry user-accepted scene/world support, while
  `track_flag == False` dense values are not valid evaluation outputs.
- Next authorized action: return to the reviewed-protocol task and freeze the
  corrected profile requirements before Layered Design.
- Blocker: none.

## Required Questions

### Dense human state

- Which tensors are dense over `(frame, prediction column)`?
- What does `track_flag` mean: observation availability, evaluation gate,
  output validity, or another state?
- At false slots, are `thetas`, `trans`, shape, scale, and derived joints
  numerically finite, temporally continuous, and semantically valid final
  reconstruction outputs?
- Does either method explicitly reconstruct through fully occluded intervals,
  and is that result serialized independently of `track_flag`?

Numerical regularity is evidence but not by itself proof of output semantics.

### Ground, camera, and scene state

- Does each artifact serialize camera intrinsics/extrinsics, ground plane,
  scale, scene transform, scene geometry/mesh/depth, or only human states?
- For every field, who produced it and at which stage: dataset GT, shared
  supplied input, method initialization, intermediate estimate, optimized
  state, or final output?
- Are Crowd4D and DyCrowd values identical/shared or independently estimated?
- Is the representation constant over the sequence and sufficient to transform
  human results into one declared scene/world coordinate?

### Association

- Freeze one prediction-column to GT-track assignment per scene/sequence using
  aggregate `channel > 0` 2D evidence and a global bipartite assignment with
  explicit unmatched/dummy support.
- Never use per-frame rematching to select easier identities.
- Keep association evidence separate from metric accuracy and from the
  GT-owned visibility/population definition.

## Inputs

- The supplied evaluation-only Crowd4D/DyCrowd package and native prediction
  artifacts.
- The accepted GT-MOT identity mappings and receipts.
- The author-relayed visibility semantics: `0` means fully/scene occluded,
  `0.5` self-occluded, and `1` unoccluded.
- Completed author-evaluator and metric-semantics audits.

## Initial Ground And Scene Inventory

The first bounded inspection covers all eight supplied `.pt` artifacts for
both methods. The probe is retained at
`tmp/2026-08-14/scratch/8c6476a7_crowd4d-dycrowd-native-scene-schema.py`.

### Crowd4D

Every artifact contains the same 20 top-level fields. Scene-relevant state
includes:

- `ground_plane (4,)` and `cam_int (4,)`;
- `extrinsic (200, 3, 4)`, which is sequence-constant to numerical noise in
  each scene (maximum within-scene deviation below `6.6e-7` in this probe);
- scalar `scene_scale_factor` and `calculated_scale`;
- `sipc (N, 3)` with `N` from 971 to 2,914 across the eight scenes, plus
  `sipc_conf`, `sipc_filled_mask`, and `ground_inlier_mask`;
- human-scene intermediates `hsip_3ds` and `hsip_error_range`.

The Crowd4D paper/project defines SIPC as the Scene Interaction Point Cloud
used with a Scene Interaction Surface (SIS) to construct HSIP. Therefore the
artifact does contain an explicit sampled scene-geometry intermediate, not
only a flat ground plane. No SIS object, dense depth, scene mesh, or complete
scene point cloud is serialized as a top-level field in the supplied artifact.
Whether SIS can be reproduced exactly from the retained SIPC and masks is not
yet established.

### DyCrowd

Every artifact contains the same 19 top-level fields. Scene-relevant state
includes:

- `ground (4,)` and `ground_plane (4,)`, which are exactly identical within
  each inspected artifact;
- `camK (3, 3)` and `cam_int (4,)`, which are exactly equivalent matrix/vector
  encodings;
- per-prediction-column `cam2prior_R` and `cam2prior_t`;
- per-segment/per-column `world2aligneds` transformation state.

No sampled scene point cloud, SIS, dense depth, or scene mesh is serialized.
The DyCrowd method source states that its scene-level camera intrinsics and
ground plane are estimated from walking/standing human priors. Its native
artifact therefore carries a ground/camera model and alignment transforms,
not an explicit reconstructed scene surface.

### Cross-method and GT checks

- Crowd4D and DyCrowd `cam_int` values differ in every scene.
- Both methods' focal lengths differ from VirtualCrowd GT in every scene;
  neither supplied camera-intrinsic vector is a direct GT copy.
- Crowd4D and DyCrowd `ground_plane` vectors differ in every scene. Direct
  comparison with the GT plane is not yet used as a provenance test because
  the human outputs and planes may be expressed in method-specific coordinate
  systems and coefficient normalization conventions.
- The supplied evaluator consumes only `cam_int` and `ground_plane` from this
  scene state. It does not consume Crowd4D `sipc`/`extrinsic` or DyCrowd
  `world2aligneds`.

Current answer to the inserted question: **Crowd4D includes a real scene
geometry intermediate (SIPC) plus ground/camera/transform state; DyCrowd
includes a scene-level ground plane and alignment transforms but no explicit
scene geometry. Neither artifact is a complete scene reconstruction export.**

The user accepted both representations as usable inputs for the corrected
protocol. Their native coordinate details remain adapter-owned rather than a
method-neutral library schema.

## Dense Human-State Audit

The all-scene bounded probe is retained at
`tmp/2026-08-14/scratch/d8b62514_dense-inactive-state-audit.py`.

### Validity layers

`track_flag` is the supplied evaluator's output-validity gate. The native
artifact also contains denser observation state:

- DyCrowd has 172,529 active person-frames. Of these, 11,290 have
  `det_j2ds_flag == False`; they are the direct artifact evidence that the
  method returns reconstructed motion without a contemporaneous 2D detection.
  This agrees with the paper's motion-prior and group-guided occlusion-recovery
  claims.
- Crowd4D has 171,091 active person-frames. Its joint-confidence proxy leaves
  138 active frames without any positive stored detection confidence.
- Neither method has a positive observation flag where `track_flag == False`.

Therefore observation availability and reconstructed-output validity are not
the same state. Occlusion recovery must be consumed through valid
`track_flag == True` outputs, not by overriding the validity gate.

### Inactive dense buffers

- Crowd4D has 5,509 inactive cells and DyCrowd has 9,271. Most are leading or
  trailing cells outside a prediction column's active lifespan; both methods
  have only 300 internal inactive cells.
- Crowd4D inactive `thetas` and `trans` are finite and nonzero, but continue to
  evolve outside the active lifespan. This regularity does not establish a
  published result.
- DyCrowd inactive state is structurally inconsistent with a final motion
  sequence: some runs repeat prior values, 351 inactive translations are all
  zero, and activation boundaries can have large pose/translation jumps.
  Even internal gaps are not uniformly reconstructed: 142 of the 300 internal
  inactive translations are zero.
- Native loaders and both supplied/HJ-composed evaluators select only
  `track_flag == True` people. No producer or evaluator contract declares the
  remaining dense buffers valid.

The inactive buffers are consequently classified as **invalid/padding for
evaluation**, even when numerically finite. A method-missing GT case must not
be silently converted into a pose by reading those cells.

## Consequences For Corrected Protocol

1. GT presence and the selected GT visibility policy own the evaluation
   denominator; a method cannot improve the denominator by dropping a case.
2. Sequence association operates on valid prediction support and permits
   explicit unmatched/dummy assignments. It never uses per-frame rematching.
3. Geometric metrics consume only valid matched outputs. Missing GT cases are
   exposed by coverage/recall and the later selected precision/F1 view rather
   than a fabricated pose or the author's fixed `150 mm` replacement.
4. DyCrowd's 11,290 active-but-unobserved outputs remain eligible when the GT
   population policy admits them; method observation flags do not replace GT
   visibility.
5. Crowd4D scene geometry and DyCrowd ground/alignment state may both feed an
   adapter-owned sequence-constant coordinate normalization. Stable evaluation
   APIs receive only normalized coordinates plus declared provenance.
6. Temporal metrics use source frame IDs and valid consecutive support; dense
   inactive padding cannot bridge a gap.

## Completion Criteria

1. Both native artifact families have a field inventory with shape, temporal
   domain, producer/provenance class, evaluator consumption, and semantic
   confidence.
2. Inactive dense human slots are classified as usable output or invalid/
   unresolved, with static and bounded numerical evidence kept distinct.
3. Every ground/scene/camera field is classified, and the task states whether a
   valid sequence-constant scene/world representation exists for each method.
4. The sequence-level association design has explicit evidence, unmatched, and
   anti-rematching semantics sufficient for later Mathematical Architecture.
5. Consequences for GT population, MPJPE/WA/W/GMPJPE, crowd-layout metrics, and
   temporal metrics are recorded without running corrected evaluation.

## Handoff

Return to the reviewed-protocol task for corrected metric/reduction
requirements. Any nontrivial reusable implementation still requires its own
Layered Design before code changes.
