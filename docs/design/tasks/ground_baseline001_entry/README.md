# ground_baseline001_entry

Layered Design record. Task of `hjlib-in-the-wild-test` campaign 02
(`campaigns/02_capability_upstreaming/task_itw_ground_estimation_from_experiments/`),
residence here because the baseline it gives a src entry to is this
repository's.

## Requirements

`baseline001` is a named, frozen ground baseline of this repository
(`docs/usage/ground_estimation.md`: shoulder-mid top, ankle-mid bottom,
confidence > 5.0, ankle / bbox width < 0.20, KDE re-weighting). Every step is
src code -- `collect_ground_observations`,
`hjlib_ground_solver.solve_ground_param_by_top_bottom_given_K`,
`hjlib_ground_solver.compute_ground_observation_kde_density` -- but the
two-pass composition (provisional plane, density weights from it, weighted
re-solve) exists only in `script/evaluate_virtualcrowd_density_balanced_rcr_ground.py`
(`evaluate_scene`; the documented baseline001 number 15.727720 m comes from
`script/evaluate_virtualcrowd_density_balanced_rcr_cartesian.py`, which calls
it with confidence 5.0) and in a `tmp/` copy that `hjlib-in-the-wild-test` used for its TJU
estimated-ground runs (zl_v3_clip11, zl_v3_clip10). User decision 2026-10-04:
it is an existing baseline, so it is absorbed into src, and nothing depends on
`tmp/`.

Must:
1. One src call from a native-pixel `Tracked_Scene` and `K` to a camera-frame
   plane, with the baseline's constants fixed in one place.
2. Reproduce the TJU runs exactly when given the same deviation they used:
   `distance_max = 200 m` in both solves (the solver default is 80 m; the
   VirtualCrowd evaluation used the default). The deviation is a parameter, and
   the caller records it.
3. Return what a run record needs: the weighted plane, the provisional
   (unweighted) plane, the observation count and the KDE effective sample size.

Not in scope: camera-up sign normalization (the in-the-wild caller already
handles the sign against its dump normal), frame subsampling (the runs used
every frame), any change to the evaluation script's behaviour.

## Code Architecture

**hjlib-evaluation**, a new sibling module `ground_estimation_baseline001.py`
(a named, frozen recipe with its own constants; `ground_estimation_protocol.py`
is 786 lines of sampling and same-ray evaluation primitives and stays as is),
re-exported from the package `__init__`:

```python
BASELINE001_TOP_JOINT_PAIR = (5, 6)
BASELINE001_BOTTOM_JOINT_PAIR = (15, 16)
BASELINE001_CONFIDENCE_THRESHOLD = 5.0             # strict >
BASELINE001_MAXIMUM_ANKLE_BBOX_WIDTH_RATIO = 0.20  # strict <
BASELINE001_H_PRIOR_M = 1.35
BASELINE001_DISTANCE_MIN_M = -5.0
BASELINE001_DISTANCE_STEP_M = 0.1
BASELINE001_DISTANCE_MAX_M = 80.0                  # the evaluation's value
BASELINE001_MINIMUM_PRE_NORMALIZATION_WEIGHT = 0.25
BASELINE001_MAXIMUM_PRE_NORMALIZATION_WEIGHT = 4.0

@dataclass(frozen=True, slots=True)
class Baseline001_Ground_Estimate:
    provisional: Ground_Estimation_Result     # unweighted RCR plane
    density: Ground_Observation_KDE_Density   # the full KDE result
    weighted: Ground_Estimation_Result        # the baseline's answer
    distance_max_m: float                     # what both solves searched

def estimate_ground_baseline001(
        tracked_scene: Tracked_Scene, K: NDArray[np.float64], *,
        distance_max_m: float = BASELINE001_DISTANCE_MAX_M,
        ) -> Baseline001_Ground_Estimate: ...
```

Every parameter the recipe depends on is passed explicitly, so a change of a
default in hjlib-ground-solver cannot silently move it: the solver closure
carries `H_prior`, `distance_min`, `distance_step`, `distance_max_m` (and the
weights on the second call); the KDE call carries both clamps. The body is
the existing composition: `collect_ground_observations` with the constants,
`estimate_ground_from_observations` (provisional), `compute_ground_observation_kde_density`
on the provisional normal, `estimate_ground_from_observations` again with the
weights. Returning the whole density keeps the arrays the evaluation script
writes, so the script can adopt the entry later without an API change (not in
this task). Dependency direction unchanged (evaluation already pins
ground-solver and detection).

**hjlib-in-the-wild-test**, `inference/ground_source.py`:

```python
ESTIMATED_GROUND_METHOD = 'hjlib-evaluation baseline001'

def estimate_ground_for_clip(
        path_scene_root: Path, name_case: str, *,
        distance_max_m: float) -> Estimated_Ground: ...
```

It resolves the clip and its tracked scene (`preprocess.tju.dump_pipeline.load_clip_and_scene`
+ `scale_tracked_scene_to_native`, the dump's own native scaling), takes K from
the clip's calibration (`preprocess.tju.calibration.read_tju_calibration`), so
a wrong K cannot be passed in, calls `estimate_ground_baseline001`, and returns
an `Estimated_Ground`. Its claim holds JSON-native values only: method,
`distance_max_m`, observation count, effective sample size, the provisional
plane as a list of floats. `read_estimated_ground` stays for result files
already written.

`inference/run_ours.py` records the ground source without assuming a file, in
a small pure function that builds the ground fields of the run record (kind,
reference path, provenance, claim) so it can be tested without a run:
`ground_reference_path` reads `ground.claim.get('source_path')`, and a
computed estimate records `None` there: the field is a path, and the method is
already in `ground_provenance` and `ground_claim` (today
`ground.claim['source_path']` would raise after every track's npz is saved,
leaving a run folder without its `run.json`).

The repository declares `hjlib-evaluation` in `[tool.hjlibm.deps]` (it is in
the closure through experiments already; src now imports it directly).

## Smoke-Test Standard

- hjlib-evaluation, on a synthetic `Tracked_Scene` with a known plane, with
  the expected values written as literals in the test (not imported):
  - observations at exactly confidence 5.0 and exactly ratio 0.20 are
    excluded, ones just past them are included;
  - through a solver spy: both calls receive `H_prior == 1.35`,
    `distance_min == -5.0`, `distance_step == 0.1` and the given
    `distance_max_m`; the first has no weights, the second has exactly the
    KDE's normalized weights; the KDE receives clamps 0.25 / 4.0;
  - the weighted plane recovers the synthetic plane within the solver's grid.
- One-off equivalence on real data (not a repository test), against
  `tmp/2026-09-26/task/est-ground-v3-clips/<clip>_crowd3d_style_result.json`:
  with `distance_max_m=200` on zl_v3_clip11 and zl_v3_clip10, after applying the `tmp/`
  script's own `camera_up` expression (sign flip and re-normalization) to the
  new planes, the weighted and provisional planes are equal
  exactly, observation counts are 1171 / 680 and effective sample sizes
  760.957 / 494.162. One known difference: the `tmp/` script dropped persons
  without bboxes, while `collect_ground_observations` raises on them; the
  clips have bboxes for every person, so equality is expected.
- hjlib-in-the-wild-test: smoke and pyright stay green; a test of the ground
  record function with a claim that has no `source_path` (reference path
  `None`, claim JSON-serializable).

## Migration Plan

1. hjlib-evaluation: module + export + test + usage doc row; commit.
2. Cascade evaluation's new HEAD leaf first: hj-tpa-crowd4d, hjlib-experiments
   (so the closure has one evaluation pin), then the rest of its consumers.
3. hjlib-in-the-wild-test: `estimate_ground_for_clip`, the `run_ours` record
   fix, the dependency declaration, usage doc; point the zl_v3 run record's
   reproduction recipe at it instead of the `tmp/` script.

## Modification History

- 2026-10-04: record written (Requirements, Code Architecture, Smoke-Test
  Standard, Migration Plan).
- 2026-10-04, Code Architecture review (read-only reviewer, round 1). Accepted
  and applied: CRITICAL -- a computed estimate has no `source_path`, and
  `run_ours` reads it unguarded after the results are written: the record now
  includes the `run_ours` change and a JSON-native claim. Concerns: pass every
  solver / KDE parameter explicitly (the recipe no longer rests on another
  repository's defaults); a smoke test that can fail on wrong constants
  (literals, boundary observations, spy on every argument); cite the Cartesian
  runner for the documented number; exact real-data equivalence with counts
  and ESS; a sibling module instead of the 786-line protocol module; return
  the whole KDE density; K from the clip calibration; evaluation re-pinned in
  experiments before this repository pins it.
- 2026-10-04, Code Architecture review round 2: every round-1 finding
  resolved, no new critical. Applied the three small ones: a computed estimate
  records `ground_reference_path = None` (the field is a path); the exact
  real-data comparison first applies the `tmp/` `camera_up` expression to the
  new planes; the run-record ground fields move into a pure function so the
  missing-`source_path` case is testable without a run. Layer accepted.
- 2026-10-04, implemented in `src/hjlib_evaluation/ground_estimation_baseline001.py`.
  Checks: the smoke test (5 cases) passes and turns red when the confidence
  threshold or `H_prior` is mutated (3 cases fail); the one-off real-data
  equivalence (`tmp/2026-10-04/scratch/3a74d426_baseline001_equivalence.py.py`)
  reproduces zl_v3_clip11 and zl_v3_clip10 bit for bit: weighted and
  provisional planes equal, observation counts 1171 / 680, ESS and objective
  equal.
