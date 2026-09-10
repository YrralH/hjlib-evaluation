'''Additive world-coordinate schema preserves native zero-based person IDs.'''
from dataclasses import replace

import numpy as np
import pytest

from hjlib_evaluation import (
    evaluate_corrected_crowd_selected_view,
    evaluate_corrected_crowd_selected_view_and_world_dynamics,
    evaluate_corrected_crowd_sequence,
    evaluate_corrected_crowd_world_dynamics,
)
from hjlib_evaluation.corrected_crowd_data import WORLD_CROWD_SCHEMA_VERSION
from test_virtualcrowd_naive_comparison import make_sequence, evaluate_result


def test_world_schema_keeps_naive_formulas_and_zero_ids() -> None:
    legacy = make_sequence(prediction_motion_scale=1.3)
    world = replace(legacy, schema_version=WORLD_CROWD_SCHEMA_VERSION,
        coordinate_frame='WORLD_METRES', gt_track_ids=np.zeros(8, dtype=np.int64),
        prediction_local_track_ids=np.zeros(8, dtype=np.int64))
    _, before = evaluate_result(legacy)
    _, after = evaluate_result(world)
    assert before == after
    with pytest.raises(ValueError):
        replace(legacy, gt_track_ids=np.zeros(8, dtype=np.int64))
    with pytest.raises(ValueError):
        replace(world, coordinate_frame='FIXED_CAMERA_WORLD_EQUIVALENT')
    with pytest.raises(ValueError):
        replace(world, gt_track_ids=np.full(8, -1, dtype=np.int64))
    selected = np.ones(8, dtype=np.bool_)
    legacy_calls = (
        lambda: evaluate_corrected_crowd_sequence(world),
        lambda: evaluate_corrected_crowd_selected_view(
            world, 'GT_VISIBLE', selected),
        lambda: evaluate_corrected_crowd_world_dynamics(
            world, 'GT_VISIBLE', selected),
        lambda: evaluate_corrected_crowd_selected_view_and_world_dynamics(
            world, 'GT_VISIBLE', selected),
    )
    for call in legacy_calls:
        with pytest.raises(ValueError, match='legacy corrected crowd'):
            call()


def smoke_test_world_crowd_schema() -> None:
    test_world_schema_keeps_naive_formulas_and_zero_ids()
