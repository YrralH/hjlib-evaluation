'''`baseline001`: the frozen person-RCR ground recipe, as one call.

`docs/usage/ground_estimation.md` names it: shoulder-mid top, ankle-mid bottom,
confidence > 5.0, ankle / bbox width < 0.20, then a provisional plane, inverse
KDE density weights computed on it, and a weighted re-solve. Every step is a
src function; this module only fixes the composition and its constants. Every
parameter the recipe depends on is passed explicitly, so a changed default in
hjlib-ground-solver cannot silently move it.
'''

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hjlib_detection import Tracked_Scene
from hjlib_ground_solver import (
    Ground_Observation_KDE_Density,
    compute_ground_observation_kde_density,
    solve_ground_param_by_top_bottom_given_K,
)

from hjlib_evaluation.ground_estimation_protocol import (
    Ground_Estimation_Result,
    Ground_Estimator,
    collect_ground_observations,
    estimate_ground_from_observations,
)


BASELINE001_TOP_JOINT_PAIR = (5, 6)
BASELINE001_BOTTOM_JOINT_PAIR = (15, 16)
BASELINE001_CONFIDENCE_THRESHOLD = 5.0             # strict >
BASELINE001_MAXIMUM_ANKLE_BBOX_WIDTH_RATIO = 0.20  # strict <
BASELINE001_H_PRIOR_M = 1.35
BASELINE001_DISTANCE_MIN_M = -5.0
BASELINE001_DISTANCE_STEP_M = 0.1
# The evaluation's value. The TJU in-the-wild runs searched to 200 m: a camera
# eight metres up a building sees people far beyond 80 m.
BASELINE001_DISTANCE_MAX_M = 80.0
BASELINE001_MINIMUM_PRE_NORMALIZATION_WEIGHT = 0.25
BASELINE001_MAXIMUM_PRE_NORMALIZATION_WEIGHT = 4.0


@dataclass(frozen=True, slots=True)
class Baseline001_Ground_Estimate:
    '''Both solves of one baseline001 run and the weights between them.

    @field provisional    : the unweighted RCR plane the density is computed on
    @field density        : the full KDE record (weights, bandwidth, ESS)
    @field weighted       : the baseline's answer
    @field distance_max_m : how far both solves searched
    '''

    provisional: Ground_Estimation_Result
    density: Ground_Observation_KDE_Density
    weighted: Ground_Estimation_Result
    distance_max_m: float


def baseline001_solver(
        distance_max_m: float,
        observation_weights: NDArray[np.float64] | None,
    ) -> Ground_Estimator:
    '''The top / bottom solver with every baseline001 parameter bound.'''
    def solve(
            top_xy_px: NDArray[np.float64],
            bottom_xy_px: NDArray[np.float64],
            K: NDArray[np.float64],
        ) -> tuple[np.ndarray, np.ndarray]:
        return solve_ground_param_by_top_bottom_given_K(
            top_xy_px,
            bottom_xy_px,
            K,
            H_prior=BASELINE001_H_PRIOR_M,
            distance_min=BASELINE001_DISTANCE_MIN_M,
            distance_max=distance_max_m,
            distance_step=BASELINE001_DISTANCE_STEP_M,
            observation_weights=observation_weights,
        )
    return solve


def estimate_ground_baseline001(
        tracked_scene: Tracked_Scene,
        K: NDArray[np.float64],
        *,
        distance_max_m: float = BASELINE001_DISTANCE_MAX_M,
    ) -> Baseline001_Ground_Estimate:
    '''Run baseline001 on a native-pixel tracked scene with intrinsics `K`.

    The plane is in the OpenCV camera frame, n.x + d = 0 with a unit normal,
    in whatever sign the solver returns.
    '''
    if not np.isfinite(distance_max_m) or distance_max_m <= 0.0:
        raise ValueError(
            'distance_max_m must be finite and positive, got %r'
            % distance_max_m)
    observations = collect_ground_observations(
        tracked_scene,
        BASELINE001_TOP_JOINT_PAIR,
        BASELINE001_BOTTOM_JOINT_PAIR,
        BASELINE001_CONFIDENCE_THRESHOLD,
        maximum_bottom_pair_bbox_width_ratio=(
            BASELINE001_MAXIMUM_ANKLE_BBOX_WIDTH_RATIO),
    )
    provisional = estimate_ground_from_observations(
        observations, K, baseline001_solver(distance_max_m, None))
    density = compute_ground_observation_kde_density(
        np.asarray(observations.bottom_xy_px, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(provisional.plane_camera_abcd[:3], dtype=np.float64),
        minimum_pre_normalization_weight=(
            BASELINE001_MINIMUM_PRE_NORMALIZATION_WEIGHT),
        maximum_pre_normalization_weight=(
            BASELINE001_MAXIMUM_PRE_NORMALIZATION_WEIGHT),
    )
    weighted = estimate_ground_from_observations(
        observations, K,
        baseline001_solver(distance_max_m,
                           density.normalized_observation_weights))
    return Baseline001_Ground_Estimate(
        provisional=provisional,
        density=density,
        weighted=weighted,
        distance_max_m=float(distance_max_m),
    )


__all__ = [
    'BASELINE001_BOTTOM_JOINT_PAIR',
    'BASELINE001_CONFIDENCE_THRESHOLD',
    'BASELINE001_DISTANCE_MAX_M',
    'BASELINE001_DISTANCE_MIN_M',
    'BASELINE001_DISTANCE_STEP_M',
    'BASELINE001_H_PRIOR_M',
    'BASELINE001_MAXIMUM_ANKLE_BBOX_WIDTH_RATIO',
    'BASELINE001_MAXIMUM_PRE_NORMALIZATION_WEIGHT',
    'BASELINE001_MINIMUM_PRE_NORMALIZATION_WEIGHT',
    'BASELINE001_TOP_JOINT_PAIR',
    'Baseline001_Ground_Estimate',
    'baseline001_solver',
    'estimate_ground_baseline001',
]
