'''Metric and support controls for paired body validation.'''
import numpy as np
import pytest
from numpy.typing import NDArray

from hjlib_evaluation.body_reconstruction_validation import (
    Body_Reconstruction_Validation_Reducer, compute_body_validation_metric_sums,
    validate_body_validation_result,
)


def bodies(count: int = 5) -> NDArray[np.float64]:
    return np.random.default_rng(10).normal(size=(count, 24, 3))


def test_alignment_translation_scale_rotation_and_reflection() -> None:
    reference = bodies()
    shifted = reference + [3., -2., 7.]
    sums = compute_body_validation_metric_sums(shifted, reference)
    assert sums.t_mpjpe_sum_m < 1e-12 and sums.pa_mpjpe_sum_m < 1e-11
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = 2.0 * (reference @ rotation.T) + [3., -2., 7.]
    sums = compute_body_validation_metric_sums(moved, reference)
    assert sums.t_mpjpe_sum_m > 1.0 and sums.pa_mpjpe_sum_m < 1e-11
    reflected = reference * [-1., 1., 1.]
    assert compute_body_validation_metric_sums(reflected, reference).pa_mpjpe_sum_m > 1.0


def test_uneven_partition_invariance_and_complete_counts() -> None:
    truth = bodies()
    predicted = truth + np.random.default_rng(11).normal(size=truth.shape) * 0.1
    whole = Body_Reconstruction_Validation_Reducer(5)
    whole.add_batch(['a', 'b', 'c', 'd', 'e'], ['bedlam'] * 5, predicted, truth, truth)
    split = Body_Reconstruction_Validation_Reducer(5)
    split.add_batch(['a', 'b', 'c', 'd'], ['bedlam'] * 4, predicted[:4], truth[:4], truth[:4])
    with pytest.raises(ValueError, match='incomplete'):
        split.finalize()
    split.add_batch(['e'], ['bedlam'], predicted[4:], truth[4:], truth[4:])
    actual, expected = split.finalize(), whole.finalize()
    for condition in ('predicted_shape', 'given_gt_shape'):
        for metric in ('t_mpjpe', 'pa_mpjpe'):
            assert actual['conditions'][condition]['metrics'][metric]['joint_count'] == 110
            assert actual['conditions'][condition]['metrics'][metric]['mean_mm'] == pytest.approx(
                expected['conditions'][condition]['metrics'][metric]['mean_mm'])


def test_duplicate_nonfinite_and_mismatched_support_fail_without_commit() -> None:
    truth = bodies(2)
    reducer = Body_Reconstruction_Validation_Reducer(2)
    with pytest.raises(ValueError, match='duplicate'):
        reducer.add_batch(['same'] * 2, ['bedlam'] * 2, truth, truth, truth)
    bad = truth.copy()
    bad[0, 8, 1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        reducer.add_batch(['a', 'b'], ['bedlam'] * 2, truth, bad, truth)
    assert not reducer.seen and not reducer.sums
    reducer.add_batch(['a', 'b'], ['bedlam'] * 2, truth, truth, truth)
    with pytest.raises(ValueError, match='duplicate'):
        reducer.add_batch(['a', 'b'], ['bedlam'] * 2, truth, truth, truth)
    assert reducer.finalize()['person_frame_count'] == 2


def smoke_test_body_reconstruction_validation() -> None:
    test_alignment_translation_scale_rotation_and_reflection()
    test_uneven_partition_invariance_and_complete_counts()
    test_duplicate_nonfinite_and_mismatched_support_fail_without_commit()
    test_result_validator_rejects_cross_condition_dataset_rename()
    test_smpl22_excludes_hands_before_fit_and_keeps_root_denominator()
    test_old_schema_cannot_validate_as_smpl22()


def test_result_validator_rejects_cross_condition_dataset_rename() -> None:
    truth = bodies(2)
    reducer = Body_Reconstruction_Validation_Reducer(2)
    reducer.add_batch(['a', 'b'], ['bedlam'] * 2, truth, truth, truth)
    result = reducer.finalize()
    validate_body_validation_result(result, 2)
    datasets = result['conditions']['given_gt_shape']['per_dataset']
    datasets['other'] = datasets.pop('bedlam')
    with pytest.raises(ValueError, match='identical per-dataset'):
        validate_body_validation_result(result, 2)


def test_smpl22_excludes_hands_before_fit_and_keeps_root_denominator() -> None:
    truth = bodies(1)
    altered = truth.copy()
    altered[:, 22:] += 10000.
    sums = compute_body_validation_metric_sums(altered, truth)
    assert sums.joint_count == 22 and sums.t_mpjpe_sum_m == 0.
    assert sums.pa_mpjpe_sum_m < 1e-11
    altered[:, 22:] = np.nan
    assert compute_body_validation_metric_sums(altered, truth).pa_mpjpe_sum_m < 1e-11
    altered = truth.copy()
    altered[0, 1, 0] += .022
    row = compute_body_validation_metric_sums(altered, truth).to_record()
    assert row['t_mpjpe']['mean_mm'] == pytest.approx(1.)


def test_old_schema_cannot_validate_as_smpl22() -> None:
    truth = bodies(1)
    reducer = Body_Reconstruction_Validation_Reducer(1)
    reducer.add_batch(['a'], ['bedlam'], truth, truth, truth)
    result = reducer.finalize()
    assert result['joint_indices'] == list(range(22))
    result['schema'] = 'body_reconstruction_validation.v1'
    with pytest.raises(ValueError, match='schema'):
        validate_body_validation_result(result, 1)
