# HJ-Composed Author-Parity Reproduction

## Requirements

### Outcome

Produce an HJ-owned, independently implemented reproduction of the two frozen
Crowd4D/DyCrowd VirtualCrowd author-evaluation workflows. The reproduction must
make author behavior inspectable and testable while preventing its legacy
choices from silently becoming the default HJ evaluation protocol.

### Evidence oracles

The comparison oracles are tracked Campaign 02 artifacts owned by
`hj-tpa-crowd4d`:

| Artifact | SHA-256 | Role |
| --- | --- | --- |
| Crowd4D fresh table | `d4043000c19e69eb274516bdc6ea7cd697e1787685cf3017c98737a96e4dfa0e` | 108 displayed numeric cells |
| DyCrowd fresh table | `e5f03929052bf490409feaa2bc0b67036634a42573f09da9bd8c8d70cc7a750b` | 108 displayed numeric cells |
| Fresh/bundled reconciliation | `e03ca36ec681d8176a8d1ca3358137b5d24ab5336edc19111543d4cab2a5f2b8` | Baseline classification |
| Paper reconciliation | `ce55b98185f5835d5488b1ff98c37a2ba159077b37018dd5b95a8ad92104d83b` | Published-row context only |

New HJ outputs and receipts must be separate tracked artifacts. They may cite
these oracles but must not modify or regenerate them in place.

### Ownership and dependency hypothesis

The preferred direction is:

```text
hj-tpa-crowd4d  ->  hjlib-evaluation and existing lower HJ owners
hjlib-evaluation  -X->  hj-tpa-crowd4d
```

`hj-tpa-crowd4d` owns native prediction parsing, the explicit author-parity
profile, Crowd4D/DyCrowd branch selection, orchestration, table reproduction,
and evidence receipts. `hjlib-evaluation` may own only reusable, method-neutral
evaluation contracts or primitives. Existing lower HJ owners retain their
established nouns; this task must reuse them rather than relocating them for
convenience.

This is a revisable architecture hypothesis, not a forced outcome. A pivot is
required if concrete API, dependency, or numerical-parity evidence shows that
it violates the actual owner boundary. Any pivot must be recorded before code
crosses the changed boundary.

### Non-negotiable boundaries

- Do not import, copy, patch, or write into the supplied evaluator residence.
- Do not claim official-upstream provenance or inference reproduction.
- Do not encode machine-local absolute paths as portable defaults; real-data
  paths are explicit run inputs.
- Do not create a dedicated environment unless demonstrated incompatibility
  requires it. Reuse `hjlib_py312` without modifying third-party material.
- Do not treat complete disk outputs as complete matched metric pairs.
- Do not replace author matching or repair behavior inside the parity profile,
  even where later review is expected to reject it.
- Do not activate the corrected-protocol task implicitly.

### Verification contract

The implementation must provide:

1. deterministic synthetic tests for each independent mathematical primitive,
   including greedy collision and missing/degenerate boundaries;
2. schema/shape/identity tests for the Crowd4D-owned native adapter;
3. an eight-scene Crowd4D run and an eight-scene DyCrowd run using external
   artifacts with a before/after supplied-residence identity check;
4. a machine-readable 216-cell comparison and human-readable tables;
5. receipts containing repository revisions, runtime/package identity, input
   identities, output hashes, profile/options, and explicit parity verdicts.

Displayed four-decimal cell equality is the primary author-parity criterion.
Raw values must also be retained wherever available so rounding cannot conceal
a numerical disagreement. Any exception blocks unconditional completion until
it is explained and classified.

## Mathematical Architecture

### Profile identity and inputs

The accepted T1 Mathematical Architecture is the execution-behavior source for
this profile. This section fixes the implementation-facing contract and does
not reinterpret legacy names as scientifically preferred semantics.

One run evaluates exactly these scene identities, in this order:

```text
scene1, scene2, scene3, scene4,
scene1_view2, scene2_view2, scene3_view2, scene4_view2
```

Each scene consumes at most the first 200 prediction frames and the first 200
decoded GT frame entries in source order. Frame positions are paired by array
position, not by validating `frame_id`. Prediction input contains
`track_flag (T,N)`, `thetas (T,N,72)`, `trans (T,N,3)`, shape-compatible
`betas` and `xscale_factor`, one intrinsic matrix, and one ground plane. The two
frozen profiles differ only in prediction root/name and temporal repair:

| Profile | Prediction | Temporal repair |
| --- | --- | --- |
| `crowd4d_author_v1` | `icml_motion_fit.pt` | most-frequent repeated ID, bounded fill |
| `dycrowd_author_v1` | `group_fit_global_v2-0.03.pt` | modal ID broadcast (`use_gt_mot=True`) |

GT is loaded from the released dataset through
`VirtualCrowd_Raw_Reader.load_scene_label()`. Campaign 01 proved its eight
label bytes equal the author-bundled copies, so using the released root changes
neither GT bytes nor decoded values and avoids a second GT loader.

### Prediction geometry

For each active prediction column, neutral SMPL runs in `float32` with ten
betas and 72 axis-angle pose values. Let `q = 1.1 ** xscale_factor`. Model-space
vertices, SMPL-54 joints, and raw H36M-17 joints are transformed as:

```text
X_camera = q * X_model + translation
```

`hjlib-smpl` supplies vertices, SMPL-54, and the H36M regressor ownership. Its
current H36M output is pelvis-centered and therefore cannot represent author
`GMPJPE`; T2 adds an owner-side forward result containing raw model-space
H36M-17 before scale/translation. The owner already fixes the regressor row
order; the TPA does not duplicate it. H36M indices `0:14` are metric joints and
index `14` is the pelvis:

```text
H14_camera = H36M17_camera[:14]
H14_local = H14_camera - H36M17_camera[14]
```

Matching projects the COCO-17 name subset of SMPL-54 camera joints with the
stored `K`; no extrinsic is applied and no positive-depth guard is inserted.
Torso center is the mean of SMPL-54 indices `[16,17,45,46]`. GT HSIP is the
author-compatible orthogonal projection of the GT torso center onto the GT
plane: first divide all four plane coefficients by the norm of its first three,
then subtract the normal-scaled signed distance. This path retains NumPy dtype
promotion and has no extra near-zero-normal guard. The existing HJ ground helper
returns forced `float32` and adds a guard, so it is not used inside the exact
compatibility profile. The computed prediction ground projection remains
unnecessary because the author metrics do not consume it.

### OKS and asymmetric greedy association

The method-neutral OKS leaf receives already-selected 2D points, positive
effective reference areas, sigmas, and a joint-valid mask. It computes:

```text
e[g,p,k] = ||pred[p,k]-gt[g,k]||^2
           / ((2*sigma[k])^2 * area[g] * 2)
OKS[g,p] = mean(exp(-e[g,p,k]) over valid k)
```

The author profile fixes:

```text
sigmas = [0.26,0.25,0.25,0.35,0.35,0.79,0.79,0.72,0.72,
          0.62,0.62,1.07,1.07,0.87,0.87,0.89,0.89] / 10
```

It passes all 17 joints as valid. For matching it supplies
`area=max(raw_bbox_area, spacing(1))`. Association exactly preserves the
author's non-global greedy rule:

1. construct `distance=1-OKS`, then take `np.min` independently for each GT;
2. process GT rows with default `np.argsort` over those minima;
3. select the first prediction where distance equals that exact minimum;
4. accept only when `minimum_distance < 1-1e-6` and the prediction is unused;
5. preserve NumPy's current version-sensitive ordering for equal GT minima;
6. a collision drops the later GT without a second-best retry.

For reported matched-pair OKS, the profile supplies
`area=max(raw_bbox_area, spacing(1)) + spacing(1)`, computes aligned pairs only,
then multiplies their mean by `M/G`. No identity from prediction `idxs` alters
this author-parity association.

### Frame metrics

For `M` accepted pairs and `G` GT people, `r=M/max(G,1)`. With matched torso
points `P` and `Q`, unordered pairs whose GT distance is strictly greater than
`np.spacing(1)` define:

```text
PPDS_pair = max(1 - abs(||P_i-P_j||-||Q_i-Q_j||)/||Q_i-Q_j||, 0)
PPDS = r * mean(PPDS_pair over all valid GT-distance pairs)
```

Fewer than two people or no valid pair returns zero.

`PA-PPDS` first applies the exact reflection-disabled
`trimesh.registration.procrustes` transform fitted from all matched torso
points, then uses the same equation. Fewer than two matches yields zero. Its
transform scale `a` gives `SS=a` for `a<=1`, else `1/(a+1e-9)`; `SS` is not
multiplied by `r`.

`PCOD` compares the sign of matched predicted torso depth differences against
matched GT HSIP depth differences. Ties are incorrect; the pair mean is
multiplied by `r`.

The public joint-error leaf returns per-joint Euclidean distances without unit
conversion or reduction policy. The TPA profile takes the mean and multiplies
metre inputs by 1000. `MPJPE` uses `H14_local`. `PA-MPJPE` fits the author's
per-person `float32` Torch SVD similarity transform before the same error.

For target points `S1 (B,N,3)` and reference points `S2 (B,N,3)`, transpose to
`(B,3,N)`, compute `mu1/2` over `N`, and center `X1/2`. Then:

```text
var1 = sum(X1**2) over coordinates and points
K = X1 @ X2^T
U, singular_values, Vh = torch.linalg.svd(K)
V = Vh^T
Z = I; Z[-1,-1] *= sign(det(U @ V^T))
R = V @ Z @ U^T
scale = trace(R @ K) / var1
t = mu2 - scale * R @ mu1
S1_aligned = scale * R @ S1 + t
```

The operation is reflection-disabled through `Z`, uses Torch `float32`, and
retains division-by-zero/non-finite behavior when `var1` degenerates. `WA` fits
one batch item after flattening all compact `(sample,joint)` points; `W` fits on
the first two compact samples and applies the same `scale/R/t` to every sample.
Each unmatched GT contributes 150 mm:

```text
frame_error = (matched_error*M + 150*(G-M))/G
```

The redundancy value and hidden composite score are preserved exactly:

```text
RP = clip(P/G - 1.02, 0, 1)
Score = 0.3*PA-PPDS + 0.2*SS + 0.1*PCOD + 0.4*OKS - 0.5*RP
```

Zero GT follows the documented author branch, although the frozen JSON schema
currently raises earlier on its rank-one empty arrays. GT with zero matches
produces 150 mm frame errors, zero positive metrics, and `Score=-0.5*RP`.
Degenerate Procrustes/SVD inputs retain the author failure/non-finite behavior;
the parity profile does not silently add a fallback.

### Temporal identity and metrics

Per-frame matches first populate prediction-column-to-GT-ID state. The Crowd4D
profile selects the most frequent non-`-1` ID occurring more than once, fills
from its first through last occurrence, and clears assignments outside that
interval; if none repeats, assignments remain. The DyCrowd profile selects the
modal ID after assigning `-1` count zero and broadcasts it across all frames.
Prediction activity subsequently gates both. Neither repair validates that the
selected GT exists in the frame.

For parity, Crowd4D obtains kinds/counts through `np.unique`, orders with
`np.argsort(counts)[::-1]`, and therefore retains version-sensitive equal-count
behavior. DyCrowd uses sorted `np.unique` plus first `np.argmax`, so the smallest
tied ID wins after the `-1` count is zeroed.

Only repaired active tracks with more than two samples enter metric numerators:

- `WA-MPJPE`: one author Torch similarity fit over all compacted
  `(sample,joint)` local points;
- `W-MPJPE`: the same fit using only the first two compacted samples, then
  applied to the entire local track;
- `GMPJPE`: unaligned absolute-camera H36M-14 error;
- `ACCEL`: compact consecutive second differences of local joints, followed by
  acceleration magnitude per joint, norm of prediction-minus-GT magnitudes
  across joints, mean over compact time, and `*1000`; no frame-gap or fps term.

The three MPJPE numerators are weighted by compact track sample count. ACCEL is
an unweighted mean over valid tracks. Every repaired active assignment,
including tracks with at most two samples, covers one unique `(GT ID, frame)`
cell for missing-count purposes. With legacy nominal person count `C`, selected
frame count `F`, numerator sample count `L`, and unique covered cells `U`:

```text
missing = C*F - U
temporal_error = (matched_error*L + 150*missing)/(L+missing)
```

If no track has more than two samples, the three temporal errors are 150 and
ACCEL is zero regardless of covered short-track cells.

### Aggregation and parity verdict

Each frame row has twelve values in the frozen order: matched ratio, PPDS,
PA-PPDS, PCOD, MPJPE, PA-MPJPE, WA-MPJPE, W-MPJPE, ACCEL, OKS, GMPJPE, Score.
Temporal scene scalars are repeated onto every frame. Scene rows are unweighted
frame means; the mean row is the mean of all concatenated frame rows. Display
uses Python four-decimal rounding and the same PrettyTable headers/order as the
oracles.

The primary verdict compares the nine displayed rows times twelve numeric
columns for each profile: 108 cells per table, 216 total. A cell is exact only
when its rendered numeric token equals the corresponding fresh-oracle token.
The receipt also records unrounded HJ values and absolute deltas. Table byte
identity is supporting evidence, not required, because formatting metadata may
be regenerated independently. A non-exact token is a failed unconditional
parity result until explicitly classified.

## Code Architecture

### Repository and dependency boundary

The reuse scan accepted the preferred direction without a pivot:

```text
hj-tpa-crowd4d
  -> hjlib-evaluation     OKS and joint-error leaves
  -> hjlib-dataset-raw    faithful released VirtualCrowd JSON reader
  -> hjlib-smpl           neutral SMPL vertices, SMPL-54, raw H36M-17
  -> hjlib-skeleton       SMPL-54 to COCO-17 name mapping
  -> hjlib-geometry       temporal second difference
```

`hjlib-evaluation` gains no TPA import, profile enum, Crowd4D schema, table
format, matching policy, or result fixture. `hj-tpa-crowd4d` becomes an ordinary
installable Python package and declares every directly imported family repo in
its package metadata and `hjlibm` roster. Leaf-first commits/pins are required:
`hjlib-evaluation` first, then the TPA and its catalog declaration.

No new `hjlib-geometry` API is needed for T2. The two author similarity
algorithms stay in the named compatibility profile because their different
dtype/backend conventions are part of parity, not a proposed generic standard.
`hjlib-smpl` does receive a narrow owner-side extension because it already owns
the H36M regressor loading, row mapping, and regression operation; the TPA must
not create a second mapping truth.

### SMPL owner extension

`hjlib-smpl` adds a frozen result type and one forward entry:

```python
@dataclass(frozen=True, slots=True)
class SMPL_Joint_Forward:
    vertices: torch.Tensor
    joints_54: torch.Tensor
    joints_h36m17: torch.Tensor


def forward_smpl_joints_from_param(
    smpl_model: SMPL_Full,
    param: SMPL_Param_Base | SMPL_Param_with_Scale_Base,
) -> SMPL_Joint_Forward: ...
```

The model must have been constructed with `J_reg_h36m17_path`. The function
performs one model-space forward, exposes the model's existing mapped raw
H36M-17 regression before centering, and then applies each param variant's
scale/translation identically to vertices and both joint sets. It preserves
Torch dtype/device/gradient and the Single/Batch/Seq leading shape convention.
The existing `forward_smpl_from_param` delegates the shared preparation and
transform path without changing its return contract.

### Stable evaluation leaves

`hjlib-evaluation` adds two stateless public modules:

```text
src/hjlib_evaluation/keypoint_oks.py
src/hjlib_evaluation/joint_error.py
```

Public functions are:

```python
def compute_keypoint_oks_matrix(
    reference_points_xy: NDArray[np.generic],
    target_points_xy: NDArray[np.generic],
    reference_areas: NDArray[np.generic],
    sigmas: NDArray[np.generic],
    reference_joint_valid: NDArray[np.bool_],
) -> NDArray[np.float64]: ...


def compute_joint_position_errors(
    target_points: NDArray[np.generic],
    reference_points: NDArray[np.generic],
) -> NDArray[np.float64]: ...
```

OKS validates `(G,K,2)`, `(P,K,2)`, `(G,)`, `(K,)`, and `(G,K)`; areas and
sigmas must be finite and strictly positive. It returns `(G,P)` and returns
zero for a GT row with no valid joints. It owns no bbox conversion, epsilon,
threshold, matching, aggregation, or visibility-source policy.

Joint error validates equal real-numeric `(...,J,3)` arrays, converts to
`float64`, and returns `(...,J)` distances. It owns no root choice, alignment,
unit conversion, missing penalty, or reduction. The existing reducer is changed
to call this public leaf so there is one MPJPE arithmetic owner. Both functions
are top-level exports and receive data-free smoke plus strict typing.

### TPA package structure

```text
src/hj_tpa_crowd4d/
  __init__.py
  author_profile.py          immutable profile/constants and bundle specs
  evaluation_data.py         typed normalized frame/track/result records
  native_prediction.py       safe path resolution, torch artifact load/validation
  prediction_geometry.py     HJ SMPL composition and raw H36M regression bridge
  author_association.py      OKS area policy and asymmetric greedy matching
  author_frame_metrics.py    PPDS/PA-PPDS/PCOD/MPJPE/score compatibility logic
  author_tracking.py         repair, temporal metrics, nominal missing penalties
  author_evaluator.py        bounded scene/frame orchestration and aggregation
  result_table.py            oracle parsing, table rendering, 216-cell comparison
  receipt.py                 input/code/runtime/output identities and JSON writing
  cli.py                     one flat Typer operation
```

No `utils` or `helpers` module is introduced. Pure functions own stateless math;
the only model-bearing object is `Crowd4D_SMPL_Adapter`, and the only workflow
object is `Author_Parity_Evaluator`. Both receive paths/profile explicitly.
Prediction and GT graphs are scoped to one scene at a time so the eight large
JSON files and prediction artifacts are not simultaneously resident.

`native_prediction.py` accepts only contained regular files below explicitly
provided prediction roots and validates all required keys/shapes before model
work. It uses Torch deserialization only on the explicitly user-supplied trusted
research artifact roots; the CLI and usage docs must state that `.pt` loading is
unsafe for untrusted files. The package never imports from the supplied
Crowd4D source tree.

`prediction_geometry.py` builds `SMPL_Full` from the exact explicit author model
file and extra regressor, calls `build_smpl_param_with_scale_batch_from_flat`
and `forward_smpl_joints_from_param`, and does not know the regressor row mapping.
It verifies the SMPL54->COCO17 mapping length is 17. Perspective projection
remains a named TPA compatibility operation: the stable HJ camera primitive is
mathematically equivalent on these intrinsics, but matrix multiplication changes
the floating-point operation order and does not preserve the frozen author
tokens. The author-compatible ground formula likewise remains TPA-local because
the existing HJ helper intentionally differs in dtype and degenerate-plane
failure policy. These are explicit parity exceptions, not proposed stable APIs.

### CLI and output transaction

The package exposes one flat Typer command:

```text
hj-tpa-crowd4d
  --path-supplied-residence <identified Crowd4D package root>
  --path-dataset-root <released VirtualCrowd root>
  --path-crowd4d-predictions <root>
  --path-dycrowd-predictions <root>
  --path-smpl-model <SMPL_NEUTRAL.pkl>
  --path-extra9-regressor <J_regressor_extra.npy>
  --path-h36m17-regressor <J_regressor_h36m.npy>
  --path-crowd4d-oracle <tracked Campaign 02 fresh table>
  --path-dycrowd-oracle <tracked Campaign 02 fresh table>
  --path-output-root <new/existing external output directory>
```

Import and `--help` perform no large-data access or model loading. Before any
output write, the command resolves and validates every selected input, verifies
the two oracle hashes against the Requirements table, and verifies that the
output root is outside all input/supplied/oracle roots. The explicit supplied
residence is the sole before/after manifest target; no common-parent inference
is permitted. Prediction/model paths must resolve inside that residence, while
the separately released dataset root and tracked oracle files must resolve
outside it.

The command captures the supplied manifest, evaluates both profiles into a
temporary sibling staging directory, captures and compares the final manifest,
writes tables/comparison/receipt, then promotes the complete bundle to the
requested output root. A failed run leaves no success receipt and never
overwrites the tracked Campaign 02 oracles.

After CLI success, the task executor absorbs the four small evidence files into
the TPA-owned tracked residence as a separate normal git-reviewed change:

```text
evidence/author_parity/
  crowd4d_summary.json
  dycrowd_summary.json
  comparison_receipt.json
  dycrowd_scene2_matched_ratio_diagnostic.json
```

The tracked comparison receipt binds the full external operation receipt and
summaries by SHA-256 while retaining the input/runtime/source and verdict
essentials needed for cold review. Large inputs, full per-frame arrays, traces,
and transient staging remain external.
The cross-owner task ledger links this evidence; it does not duplicate it in
`hjlib-evaluation`.

Tracked absorption is deliberately not a second runtime transaction. The CLI
never writes a repository. Before adding/replacing tracked files, the executor
checks that all four external hashes agree with the successful receipt; the
ordinary working-tree diff and commit gate then make partial or unintended
tracked changes visible. Existing tracked evidence is never overwritten by an
incomplete or failed run. If a later rerun intentionally replaces it, snapshot
and diff review apply like any other repository edit.

### Test placement and gates

- `hjlib-evaluation/test_smoke/`: OKS shapes/masks/known values/empty GT or
  prediction axes; joint-error known values, dtype, shape, non-finite policy;
  existing reducer regression and master runner.
- `hj-tpa-crowd4d/test_smoke/`: synthetic native schema, H36M absolute/local
  split, greedy collision/no-second-choice case, both repair modes, short-track
  missing coverage, PPDS/PCOD/legacy ACCEL, aggregation/rounding, CLI help and
  output-transaction failure.
- `hjlib-smpl/test_smoke/`: raw H36M model-space output, identical affine
  scale/translation application across vertices/joints, variant shapes, missing
  regressor failure, and unchanged legacy forward outputs.
- `hj-tpa-crowd4d/test/`: configured real-data operation that runs both full
  profiles and asserts the tracked 216-cell comparison. Explicit CLI paths are
  also the canonical evidence-generation path and do not require modifying a
  shared local setting.

Every modified Python repository runs focused/full pytest as applicable and
strict Pyright with the `hjlib_py312` Python path. The final real-data operation
runs a supplied-residence before/after manifest check and fails if identity
changes.

## Smoke-Test Standard

The data-free and real-data split is fixed by Code Architecture. Synthetic
smoke must independently prove the collision behavior, two identity-repair
modes, unmatched penalties, alignment dtype convention, compact-time ACCEL,
nominal-slot temporal reduction, and 4-decimal token comparison rather than
only exercising an end-to-end happy path.

The real-data gate is complete only when:

1. every recorded input hash equals the frozen receipt identity;
2. the released label hashes equal the author-bundle hashes already frozen by
   Campaign 01;
3. both eight-scene runs finish and comparison accounts for 216/216 cells;
4. every non-exact HJ token is classified with raw evidence; an exact profile
   is labeled exact, while a classified drift is never labeled parity success;
5. no input/supplied file content or behavior-relevant metadata changed;
6. tracked evidence hashes match the promoted external run artifacts.

If one scene fails, the operation may stop without running later scenes, as the
user permitted for full testing. The receipt must identify the first incomplete
scene/profile and must not report a parity success.

## Modification History

- 2026-08-12: Created the task-specific Layered Design residence. Recorded the
  author-table identities, provisional TPA-to-library dependency direction,
  immutable parity boundary, and separate user-gated corrected-protocol task.
- 2026-08-12: Completed the initial Mathematical and Code Architecture drafts
  after a family-wide reuse scan. Kept the selected TPA-to-library direction,
  reused the existing VirtualCrowd reader instead of adding a GT loader, and
  isolated author-only matching/alignment/reduction from stable evaluation
  leaves. Both substantial layers remain pending dedicated review.
- 2026-08-12: Initial Mathematical Architecture review found three Critical
  fidelity gaps and four Concerns. Fixed the exact Torch similarity equations,
  PPDS epsilon boundary, OKS sigma vector, association/repair tie operations,
  H36M owner/transform order, and author-ground compatibility boundary. Initial
  boundary review separately found one Critical stale TPA entrance contract and
  the same H36M second-truth concern; both dispositions are accepted.
- 2026-08-12: Focused Mathematical re-review accepted all seven dispositions
  with no new Critical; Mathematical Architecture is accepted. Focused boundary
  re-review accepted both prior dispositions with no new Critical. Corrected its
  sole non-blocking stale dependency label before Code Architecture review.
- 2026-08-12: Code Architecture review found two Critical missing input
  contracts and one output-transaction Concern. Added explicit supplied-
  residence and two hash-verified oracle paths, made the CLI a single external
  evidence transaction, and separated tracked absorption into normal git review.
- 2026-08-12: Implemented and ran the HJ-composed profiles. Full comparison is
  complete for 216/216 cells with identical supplied-residence manifests.
  Crowd4D is exact in 108/108. DyCrowd is exact in 107/108; its sole
  `scene2 / matched ratio` drift is classified by identical HJ/current-author
  200-frame populations and raw mean, not hidden as parity success.
- 2026-08-12: Implementation review found four Critical transaction/comparison
  defects. Fixed atomic sibling staging and promotion, pre-write containment,
  explicit comparison-complete versus parity-exact verdicts, and mismatch
  serialization; removed duplicate/dead execution and camera paths and expanded
  the synthetic transaction gates before focused re-review.
- 2026-08-12: First focused re-review accepted those fixes but found two further
  Critical gaps in prediction preflight and result artifacts; both were fixed.
  Second focused re-review accepted them and found one final raw-value evidence
  gap. After adding raw means and complete 216-cell delta records, the user
  explicitly authorized a third focused review; it accepted the disposition
  with no new Critical. Mathematical Architecture, Code Architecture, and the
  implementation are accepted, and T2 is complete.
