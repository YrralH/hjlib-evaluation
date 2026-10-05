'''VRv1 camera selection, real codec IO, scene-frame GT slicing and WP parity.'''
from pathlib import Path

import numpy as np
import pytest

from hjlib_dataset_assembly.single_seq.label import Single_Seq_Label
from hjlib_dataset_assembly.single_seq.label_codec import dump_label
from hjlib_dataset_assembly import VRV1_GOPRO_SCENES
from hjlib_evaluation import (
    Dumped_SMPL_GT_Provider, VRv1_TestSet_Builder,
    VRV1_EVAL_META, VRV1_GOPRO_EVAL_META, get_gt_provider, get_testset_builder,
)
from hjlib_evaluation.per_dataset.gt_provider_wp import WP_GT_Provider
from hjlib_evaluation.per_dataset.wp_eval_meta import WP_EVAL_META


def make_full_label(count: int, scene_start: int = 0) -> Single_Seq_Label:
    frames = np.arange(count, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    camera = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    transforms = np.repeat(identity[None], count, axis=0)
    return Single_Seq_Label(
        num_frame_seq=count,
        camera_K=camera,
        camera_extrinsics=transforms,
        ground_param_world=np.array([0, 1, 0, 0], dtype=np.float32),
        image_size=np.array([256, 256]),
        camera_K_original=camera,
        camera_pose_local_relative_to_original=transforms,
        image_size_original=np.array([1920, 1080]),
        hvip_3D=np.zeros((count, 3), dtype=np.float32),
        hvip_2D=np.zeros((count, 2), dtype=np.float32),
        keypoints_2D=np.zeros((count, 17, 3), dtype=np.float32),
        bbox_2D=np.zeros((count, 4), dtype=np.float32),
        smpl_param_world={
            'poses': np.repeat(frames[:, None], 72, axis=1),
            'shapes': np.repeat(frames[:, None], 10, axis=1),
            'trans': np.repeat(frames[:, None], 3, axis=1),
        },
        smpl_joints_54_world=np.broadcast_to(
            frames[:, None, None], (count, 54, 3)).copy(),
        index_original_multi_person_seq_start=scene_start,
        index_original_multi_person_seq_end=scene_start + count,
    )


def write_label(root: Path, scene: str, label: Single_Seq_Label) -> None:
    folder = root / ('%s_single_label' % scene)
    folder.mkdir(parents=True, exist_ok=True)
    dump_label(label, str(folder / '0000_0000.bin'))


def test_main_four_whole_runs_and_gt_slice(tmp_path: Path) -> None:
    root = tmp_path / 'vrv1_smpl_fitted'
    scenes = ('S21_0', 'S22_0', 'S23_0', 'S24_0')
    counts = (1050, 1200, 1650, 1200)
    for scene, count in zip(scenes, counts, strict=True):
        write_label(root, scene, make_full_label(count, scene_start=11))
    (root / 'test.txt').write_text('\n'.join(scenes) + '\n')
    builder = get_testset_builder('vrv1_smpl_fitted', str(tmp_path))
    assert isinstance(builder, VRv1_TestSet_Builder)
    testset = builder.build('full', 'test')
    assert testset.fps == 30.0
    assert tuple(segment.name_scene for segment in testset.test_segments) == scenes
    assert len(testset) == len(testset.divider) == 4
    assert testset.filter_stats is not None
    assert testset.filter_stats.total_frames == 5100
    assert testset.filter_stats.bias_dropped_count == 0
    for index, count in enumerate(counts):
        info = testset.divider.get_seq_info(index)
        assert (info.index_within_singleseq_start, info.index_within_singleseq_end) == (0, count)
        segment = testset.get_test_segment(index)
        assert (segment.index_frame_original_start, segment.index_frame_original_end) == (11, 11 + count)
    gt = get_gt_provider('vrv1_smpl_fitted', str(tmp_path))
    assert isinstance(gt, Dumped_SMPL_GT_Provider)
    assert gt.get_eval_meta() is VRV1_EVAL_META
    expected = make_full_label(1050)
    np.testing.assert_array_equal(
        gt.get_smpl_joints_54_world('S21_0', '0000_0000', (13, 16)),
        expected.smpl_joints_54_world[2:5])
    params = gt.get_smpl_param_world('S21_0', '0000_0000', (13, 16))
    for key in ('poses', 'shapes', 'trans'):
        np.testing.assert_array_equal(params[key], expected.smpl_param_world[key][2:5])
    with pytest.raises(AssertionError):
        gt.get_smpl_joints_54_world('S21_0', '0000_0000', (10, 13))


def test_explicit_gopro_is_separate_and_50_fps(tmp_path: Path) -> None:
    root = tmp_path / 'vrv1_gopro_smpl_fitted'
    for scene in VRV1_GOPRO_SCENES:
        write_label(root, scene, make_full_label(9, 40))
    builder = get_testset_builder('vrv1_gopro_smpl_fitted', str(tmp_path))
    assert isinstance(builder, VRv1_TestSet_Builder)
    testset = builder.build('full', 'all')
    assert testset.fps == 50.0
    assert len(testset) == 20
    assert tuple(segment.name_scene for segment in testset.test_segments) == VRV1_GOPRO_SCENES
    assert testset.get_test_segment(0).index_frame_original_start == 40
    assert get_gt_provider('vrv1_gopro_smpl_fitted', str(tmp_path)).get_eval_meta() is VRV1_GOPRO_EVAL_META


@pytest.mark.parametrize('dataset,scene', [
    ('vrv1_smpl_fitted', 'S21_1072'),
    ('vrv1_gopro_smpl_fitted', 'S21_0'),
])
def test_cross_view_roots_are_rejected(tmp_path: Path, dataset: str, scene: str) -> None:
    root = tmp_path / dataset
    write_label(root, scene, make_full_label(3))
    own_scene = 'S21_0' if dataset == 'vrv1_smpl_fitted' else 'S21_1072'
    write_label(root, own_scene, make_full_label(3))
    (root / 'test.txt').write_text(own_scene + '\n')
    with pytest.raises(ValueError, match='other camera family'):
        get_testset_builder(dataset, str(tmp_path)).build('full', 'test')


def test_policy_fps_and_metric_contract(tmp_path: Path) -> None:
    builder = get_testset_builder('vrv1_smpl_fitted', str(tmp_path))
    with pytest.raises(ValueError, match='only full'):
        builder.build('visualize', 'all')
    with pytest.raises(ValueError, match='FPS differ'):
        VRv1_TestSet_Builder('vrv1_smpl_fitted', str(tmp_path), 50.0)
    with pytest.raises(ValueError, match='vrv1_smpl_fitted'):
        get_testset_builder('vrv1', str(tmp_path))
    with pytest.raises(ValueError, match='vrv1_smpl_fitted'):
        get_gt_provider('vrv1', str(tmp_path))
    for meta in (VRV1_EVAL_META, VRV1_GOPRO_EVAL_META):
        assert meta.meta_version == '2026-05-20_v1'
        assert meta.unit_world == 'm' and meta.k_rt_relation == 'shared'
        assert meta.metrics_2d_oks == ()
        assert len(meta.metrics_3d) == 1
        metric = meta.metrics_3d[0]
        assert metric.name == 'SMPL_24_full'
        assert metric.joint_indices_smpl_54 == tuple(range(24))
        assert metric.root_indices_smpl_54_for_alignment == (0,)


def test_wp_wrapper_preserves_metadata_and_slicing(tmp_path: Path) -> None:
    root = tmp_path / 'worldpose'
    label = make_full_label(8, 101)
    write_label(root, 'wp_scene', label)
    provider = get_gt_provider('worldpose_smpl', str(tmp_path))
    assert isinstance(provider, WP_GT_Provider)
    assert provider.name_dataset == 'worldpose_smpl'
    assert provider.get_eval_meta() is WP_EVAL_META
    np.testing.assert_array_equal(
        provider.get_smpl_joints_54_world('wp_scene', '0000_0000', (104, 107)),
        label.smpl_joints_54_world[3:6])
    for key, value in provider.get_smpl_param_world('wp_scene', '0000_0000', (104, 107)).items():
        np.testing.assert_array_equal(value, label.smpl_param_world[key][3:6])
    with pytest.raises(ValueError, match='metric identity'):
        Dumped_SMPL_GT_Provider('vrv1_smpl_fitted', str(root), WP_EVAL_META)
