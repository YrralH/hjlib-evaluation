'''Smoke: the testset layer's data-free units. No external data.

Covers Test_Segment, the TestSet container (len / get_test_segment /
restrict_to_scenes alignment / summary), and get_testset_builder path resolution +
error paths. The full end-to-end build (real washed filter store + dumped labels)
lives in test/ (FAIL-not-skip), since it needs real data.
'''
import os.path as osp
import sys

from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider

from hjlib_evaluation import (
    Filter_Stats,
    Test_Segment,
    TestSet,
    get_testset_builder,
)


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


def _make_filter_stats(test_segment_count: int, total_frames: int) -> Filter_Stats:
    return Filter_Stats(
        bias_config_name=None, bias_tag=None, min_bias_segment_length=None,
        n_frame_min_seq=None, produced_at=None,
        raw_seq_count=3, bias_dropped_count=1, short_dropped_count=0, kept_seq_count=2,
        test_segment_count=test_segment_count, shortest_segment=30, longest_segment=40,
        total_frames=total_frames,
    )


def _make_testset() -> TestSet:
    # Two scenes, three kept sub-segments. Divider ranges (seq-local) are built in
    # lockstep with the scene-level Test_Segments (scene offset baked in).
    ranges = [
        ('scene_A', '0000_0000', 0, 40),
        ('scene_A', '0000_0000', 60, 90),
        ('scene_B', '0002_0000', 0, 30),
    ]
    segments = [
        Test_Segment('worldpose_smpl', 'scene_A', '0000_0000', 0, 1000 + 0,  1000 + 40),
        Test_Segment('worldpose_smpl', 'scene_A', '0000_0000', 0, 1000 + 60, 1000 + 90),
        Test_Segment('worldpose_smpl', 'scene_B', '0002_0000', 2, 5000 + 0,  5000 + 30),
    ]
    return TestSet(
        name_dataset='worldpose_smpl', policy='full', split='all',
        divider=Filtered_Sub_Seq_Divider(ranges), test_segments=segments,
        path_root_label='/dump/worldpose', fps=50.0,
        filter_stats=_make_filter_stats(3, 40 + 30 + 30),
    )


def _check_test_segment() -> None:
    seg = Test_Segment('jta_smpl_fitted', 'scene_X', '0003_0001', 3, 100, 140)
    assert seg.length == 40, seg.length
    assert 'scene_X' in seg.to_str() and 'pid=3' in seg.to_str(), seg.to_str()


def _check_testset_container() -> None:
    testset = _make_testset()
    assert len(testset) == 3, len(testset)
    assert len(testset) == len(testset.divider), (len(testset), len(testset.divider))
    seg = testset.get_test_segment(1)
    assert seg.index_frame_original_start == 1060, seg.index_frame_original_start
    info = testset.divider.get_seq_info(1)
    assert (info.name_scene, info.index_within_singleseq_start) == ('scene_A', 60), info
    text = testset.summary()
    assert 'dataset=worldpose_smpl' in text and 'grand total' in text, text


def _check_restrict_to_scenes() -> None:
    testset = _make_testset()
    testset.restrict_to_scenes(['scene_A'])
    assert len(testset) == 2, len(testset)
    assert len(testset.divider) == len(testset.test_segments), (
        len(testset.divider), len(testset.test_segments))
    assert all(s.name_scene == 'scene_A' for s in testset.test_segments)
    # divider stays index-aligned with the restricted segments
    for i in range(len(testset)):
        assert testset.divider.get_seq_info(i).name_scene == 'scene_A'
    assert testset.filter_stats is not None
    assert testset.filter_stats.test_segment_count == 2, testset.filter_stats.test_segment_count


def _check_get_testset_builder_paths() -> None:
    b_wp = get_testset_builder('worldpose_smpl', path_dump_root='/dump', path_filter_stats_base='/fs')
    assert b_wp.path_root_label == '/dump/worldpose', b_wp.path_root_label
    assert b_wp.path_filter_store == '/fs/wp_filter_stats/seq_modifications_jsonbin', b_wp.path_filter_store
    assert b_wp.fps == 50.0, b_wp.fps

    b_jta = get_testset_builder('jta_smpl_fitted', path_dump_root='/dump', path_filter_stats_base='/fs')
    assert b_jta.path_root_label == '/dump/jta_smpl_fitted', b_jta.path_root_label
    assert b_jta.path_filter_store == '/fs/jta_filter_stats/seq_modifications_jsonbin', b_jta.path_filter_store

    b_ext = get_testset_builder('jta_ext_smpl_fitted', path_dump_root='/dump', path_filter_stats_base='/fs')
    assert b_ext.path_root_label == '/dump/jta_ext_smpl_fitted', b_ext.path_root_label
    assert b_ext.path_filter_store == '/fs/jta_ext_filter_stats/seq_modifications_jsonbin', b_ext.path_filter_store


def _check_error_paths() -> None:
    try:
        get_testset_builder('vrv1', path_dump_root='/dump', path_filter_stats_base='/fs')
    except NotImplementedError:
        pass
    else:
        raise AssertionError('vrv1 should raise NotImplementedError')

    try:
        get_testset_builder('nope', path_dump_root='/dump', path_filter_stats_base='/fs')
    except AssertionError:
        pass
    else:
        raise AssertionError('unknown dataset should assert')

    builder = get_testset_builder('jta_smpl_fitted', path_dump_root='/dump', path_filter_stats_base='/fs')
    try:
        builder.build(policy='visualize_v3', split='all')
    except NotImplementedError:
        pass
    else:
        raise AssertionError('curated policy should raise NotImplementedError')


def test_test_segment() -> None:
    _check_test_segment()


def test_testset_container() -> None:
    _check_testset_container()


def test_restrict_to_scenes() -> None:
    _check_restrict_to_scenes()


def test_get_testset_builder_paths() -> None:
    _check_get_testset_builder_paths()


def test_error_paths() -> None:
    _check_error_paths()


def smoke_test_testset() -> None:
    _check_test_segment()
    _check_testset_container()
    _check_restrict_to_scenes()
    _check_get_testset_builder_paths()
    _check_error_paths()
    print('smoke_test_testset: all checks passed')


if __name__ == '__main__':
    smoke_test_testset()
