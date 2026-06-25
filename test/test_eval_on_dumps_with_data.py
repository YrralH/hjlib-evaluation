'''Real-data test: run stage_eval on real monolith inference dumps.

The cleanest eval-protocol validation: the monolith's per-segment prediction dumps are
real, monolith-identical predictions; reducing them with the new eval (qualname-routed
read + real GT + the ported MPJPE / T-MPJPE / Jitter math) exercises the entire
eval-on-dumps path. FAIL-not-skip: needs the dump base + dataset roots in
test/local_setting_test.py.

Auto-discovers a worldpose/full exp_tag dump dir and the split that fully covers it,
then runs eval_dumps_against_gt (whose internal asserts -- pkl present, segment match,
shape match, non-NaN at meta indices -- ARE the validation).
'''
import glob
import os.path as osp
import sys


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

try:
    from local_setting_test import (
        PATH_DUMP_ROOT_HJLIB,
        PATH_FILTER_STATS_BASE,
        PATH_INFERENCE_DUMPS_BASE,
    )
except ImportError as exc:  # FAIL-not-skip
    raise AssertionError(
        'set PATH_DUMP_ROOT_HJLIB + PATH_FILTER_STATS_BASE + PATH_INFERENCE_DUMPS_BASE '
        'in test/local_setting_test.py to run the eval-on-dumps test') from exc

from typing import Set

from hjlib_evaluation import (
    TestSet,
    build_segment_tag,
    eval_dumps_against_gt,
    get_gt_provider,
    get_testset_builder,
    list_dump_segment_tags,
    path_pkl_for_segment,
)


def _testset_tags(testset: TestSet) -> Set[str]:
    return {
        build_segment_tag(s.name_scene, s.name_seq, s.id_person,
                          s.index_frame_original_start, s.index_frame_original_end)
        for s in testset.test_segments
    }


def _check_eval_on_worldpose_dump() -> None:
    # The on-disk monolith inference-dump tree is keyed by the BARE dataset dir name
    # (this is a filesystem path, not a registry lookup -- stays 'worldpose').
    exp_dirs = sorted(glob.glob(
        osp.join(PATH_INFERENCE_DUMPS_BASE, 'worldpose', 'full', 'kp_*', '*')))
    exp_dirs = [d for d in exp_dirs if list_dump_segment_tags(d)]
    assert exp_dirs, (
        'no worldpose/full inference-dump dirs with pkls under %s' % PATH_INFERENCE_DUMPS_BASE)
    dump_dir = exp_dirs[0]
    dump_tags = set(list_dump_segment_tags(dump_dir))

    builder = get_testset_builder(
        'worldpose_smpl', path_dump_root=PATH_DUMP_ROOT_HJLIB,
        path_filter_stats_base=PATH_FILTER_STATS_BASE)
    for split in ('test', 'all', 'train', 'val'):
        testset = builder.build(policy='full', split=split)
        if len(testset) > 0 and not (_testset_tags(testset) - dump_tags):
            gt = get_gt_provider('worldpose_smpl', path_dump_root=PATH_DUMP_ROOT_HJLIB)
            # Internal asserts (pkl present / segment match / shape / non-NaN at meta
            # indices) are the validation; completing without raising = the path works.
            eval_dumps_against_gt(testset, gt, dump_dir, path_pkl_for_segment)
            print('eval-on-dumps OK: worldpose/full split=%s, %d segments, dump=%s'
                  % (split, len(testset), osp.basename(dump_dir)))
            return
    raise AssertionError(
        'no worldpose split fully covered by dump %s (testset/dump tag mismatch)' % dump_dir)


def test_eval_on_worldpose_dump() -> None:
    _check_eval_on_worldpose_dump()


def smoke_test_eval_on_dumps_with_data() -> None:
    _check_eval_on_worldpose_dump()
    print('smoke_test_eval_on_dumps_with_data: all checks passed')


if __name__ == '__main__':
    smoke_test_eval_on_dumps_with_data()
