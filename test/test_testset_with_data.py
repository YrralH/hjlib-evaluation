'''Real-data test: end-to-end TestSet build + assembly construction.

Builds a TestSet from the real washed filter store + dumped labels, then routes its
divider through build_test_assembly. FAIL-not-skip (family policy): if the roots are
not configured in test/local_setting_test.py, this raises AssertionError with a hint.

Validates: non-empty testset, divider <-> test_segment index alignment + the
seq-local -> scene-level offset (divider seq-local + scene offset == segment
scene-level coords), and that the assembly Dataset length matches the testset.
'''
import os.path as osp
import sys


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

try:
    from local_setting_test import (
        PATH_DUMP_ROOT_HJLIB,
        PATH_FILTER_STATS_BASE,
        PATH_JTA_EXT_DATA_ROOT,
    )
except ImportError as exc:  # FAIL-not-skip
    raise AssertionError(
        'set PATH_DUMP_ROOT_HJLIB + PATH_FILTER_STATS_BASE + PATH_JTA_EXT_DATA_ROOT in '
        'test/local_setting_test.py (dump root + filter-stats base + raw JTA-Ext root) '
        'to run the data-dependent test/ tree') from exc

import numpy as np

from hjlib_evaluation import build_test_assembly, get_gt_provider, get_testset_builder


def _check_build_and_align(name_dataset: str, split: str) -> None:
    builder = get_testset_builder(
        name_dataset,
        path_dump_root=PATH_DUMP_ROOT_HJLIB,
        path_filter_stats_base=PATH_FILTER_STATS_BASE,
    )
    testset = builder.build(policy='full', split=split)
    assert len(testset) > 0, (name_dataset, split, len(testset))
    assert len(testset) == len(testset.divider) == len(testset.test_segments), (
        len(testset), len(testset.divider), len(testset.test_segments))

    # Index alignment + the seq-local -> scene-level offset, checked per item.
    for index in range(len(testset)):
        info = testset.divider.get_seq_info(index)
        seg = testset.get_test_segment(index)
        assert seg.name_scene == info.name_scene, (index, seg.name_scene, info.name_scene)
        assert seg.name_seq == info.name_seq, (index, seg.name_seq, info.name_seq)
        # both ends share ONE scene offset (start - within_start == end - within_end)
        offset_start = seg.index_frame_original_start - info.index_within_singleseq_start
        offset_end = seg.index_frame_original_end - info.index_within_singleseq_end
        assert offset_start == offset_end, (index, offset_start, offset_end)
        assert seg.index_frame_original_end > seg.index_frame_original_start, index

    # Route the divider through the assembly factory (label-only path, no encoder).
    dataset = build_test_assembly(testset, flag_test_mode=True)
    assert len(dataset) == len(testset), (len(dataset), len(testset))
    print('%s/%s: %d test segments across %d scenes; assembly len=%d'
          % (name_dataset, split, len(testset),
             len(set(s.name_scene for s in testset.test_segments)), len(dataset)))


def _check_gt_worldpose() -> None:
    testset = get_testset_builder(
        'worldpose_smpl', path_dump_root=PATH_DUMP_ROOT_HJLIB,
        path_filter_stats_base=PATH_FILTER_STATS_BASE,
    ).build(policy='full', split='test')
    gt = get_gt_provider('worldpose_smpl', path_dump_root=PATH_DUMP_ROOT_HJLIB)
    seg = testset.get_test_segment(0)
    rng = (seg.index_frame_original_start, seg.index_frame_original_end)
    joints = gt.get_smpl_joints_54_world(seg.name_scene, seg.name_seq, rng)
    assert joints.shape == (seg.length, 54, 3), joints.shape
    # WP GT is a dense SMPL fit -> the SMPL_24 metric slots are non-NaN.
    meta_idx = gt.get_eval_meta().metrics_3d[0].joint_indices_smpl_54
    assert not np.isnan(joints[:, list(meta_idx), :]).any(), 'WP metric joints must be non-NaN'
    param = gt.get_smpl_param_world(seg.name_scene, seg.name_seq, rng)
    assert param['poses'].shape == (seg.length, 72), param['poses'].shape
    assert param['shapes'].shape == (seg.length, 10), param['shapes'].shape
    assert param['trans'].shape == (seg.length, 3), param['trans'].shape
    print('worldpose GT: joints %s + param ok (seg0 len=%d)' % (joints.shape, seg.length))


def _check_gt_jta_ext() -> None:
    assert PATH_JTA_EXT_DATA_ROOT is not None, (
        'set PATH_JTA_EXT_DATA_ROOT in test/local_setting_test.py for jta_ext GT')
    testset = get_testset_builder(
        'jta_ext_smpl_fitted', path_dump_root=PATH_DUMP_ROOT_HJLIB,
        path_filter_stats_base=PATH_FILTER_STATS_BASE,
    ).build(policy='full', split='test')
    gt = get_gt_provider('jta_ext_smpl_fitted', path_dump_root=PATH_DUMP_ROOT_HJLIB,
                         path_raw_data_root=PATH_JTA_EXT_DATA_ROOT)
    seg = testset.get_test_segment(0)
    rng = (seg.index_frame_original_start, seg.index_frame_original_end)
    joints = gt.get_smpl_joints_54_world(seg.name_scene, seg.name_seq, rng)
    assert joints.shape == (seg.length, 54, 3), joints.shape
    # the 12 limb-endpoint slots (with_hip metric) are populated, non-NaN.
    meta = gt.get_eval_meta()
    with_hip = {m.name: m for m in meta.metrics_3d}['limb_endpoints_with_hip']
    assert not np.isnan(joints[:, list(with_hip.joint_indices_smpl_54), :]).any(), \
        'JTA limb-endpoint joints must be non-NaN'
    # JTA has no SMPL fit -> param raises.
    try:
        gt.get_smpl_param_world(seg.name_scene, seg.name_seq, rng)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('jta_ext get_smpl_param_world should raise')
    print('jta_ext GT: joints %s + 12 limb slots non-NaN + param raises (seg0 len=%d)'
          % (joints.shape, seg.length))


def test_build_worldpose_test() -> None:
    _check_build_and_align('worldpose_smpl', 'test')


def test_build_jta_ext_test() -> None:
    _check_build_and_align('jta_ext_smpl_fitted', 'test')


def test_gt_worldpose() -> None:
    _check_gt_worldpose()


def test_gt_jta_ext() -> None:
    _check_gt_jta_ext()


def smoke_test_testset_with_data() -> None:
    _check_build_and_align('worldpose_smpl', 'test')
    _check_build_and_align('jta_ext_smpl_fitted', 'test')
    _check_gt_worldpose()
    _check_gt_jta_ext()
    print('smoke_test_testset_with_data: all checks passed')


if __name__ == '__main__':
    smoke_test_testset_with_data()
