from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider

from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import Filter_Stats, TestSet


def build_testset() -> TestSet:
    divider = Filtered_Sub_Seq_Divider([
        ('scene_a', 'seq_a', 10, 390),
        ('scene_b', 'seq_b', 5, 90),
    ])
    segments = [
        Test_Segment(
            name_dataset='dummy_smpl',
            name_scene='scene_a',
            name_seq='seq_a',
            id_person=7,
            index_frame_original_start=1000,
            index_frame_original_end=1380,
        ),
        Test_Segment(
            name_dataset='dummy_smpl',
            name_scene='scene_b',
            name_seq='seq_b',
            id_person=9,
            index_frame_original_start=205,
            index_frame_original_end=290,
        ),
    ]
    filter_stats = Filter_Stats(
        bias_config_name='dummy',
        bias_tag='dummy',
        min_bias_segment_length=1,
        n_frame_min_seq=1,
        produced_at='now',
        raw_seq_count=2,
        bias_dropped_count=0,
        short_dropped_count=0,
        kept_seq_count=2,
        test_segment_count=2,
        shortest_segment=85,
        longest_segment=380,
        total_frames=465,
    )
    return TestSet(
        name_dataset='dummy_smpl',
        policy='full',
        split='test',
        divider=divider,
        test_segments=segments,
        path_root_label='/tmp/unused',
        fps=30.0,
        filter_stats=filter_stats,
    )


def main() -> None:
    original = build_testset()
    subset = original.make_fixed_window_testset(
        length_window=120,
        length_overlap=0,
        tail_policy='drop',
    )

    assert len(original) == 2, len(original)
    assert len(subset) == 3, len(subset)
    assert subset.divider.get_seq_info(0).index_within_singleseq_start == 10
    assert subset.divider.get_seq_info(0).index_within_singleseq_end == 130
    assert subset.divider.get_seq_info(1).index_within_singleseq_start == 130
    assert subset.divider.get_seq_info(1).index_within_singleseq_end == 250
    assert subset.divider.get_seq_info(2).index_within_singleseq_start == 250
    assert subset.divider.get_seq_info(2).index_within_singleseq_end == 370

    first_segment = subset.get_test_segment(0)
    assert first_segment.id_person == 7
    assert first_segment.index_frame_original_start == 1000
    assert first_segment.index_frame_original_end == 1120
    third_segment = subset.get_test_segment(2)
    assert third_segment.index_frame_original_start == 1240
    assert third_segment.index_frame_original_end == 1360

    assert subset.filter_stats is not None
    assert subset.filter_stats.test_segment_count == 3
    assert subset.filter_stats.shortest_segment == 120
    assert subset.filter_stats.longest_segment == 120
    assert subset.filter_stats.total_frames == 360

    try:
        original.make_fixed_window_testset(length_window=120, length_overlap=120)
    except ValueError:
        pass
    else:
        raise AssertionError('length_overlap >= length_window should raise')

    try:
        original.make_fixed_window_testset(length_window=120, tail_policy='pad')
    except ValueError:
        pass
    else:
        raise AssertionError('tail_policy != drop should raise')

    print('[PASS] test_testset_fixed_window')


if __name__ == '__main__':
    main()
