'''Tester: dataset-agnostic eval driver.

Port of monolith ``test/protocol_dynamic/tester.py``. Stages:
    stage_list_segments : traverse TestSet + Assembly (+ optional GT-joints pull) for
                          wiring sanity.
    stage_inference     : run driver.infer() per segment, dump one per-segment pkl
                          ({'segment': Test_Segment, 'pred': <driver output>}).
                          Requires a live Network_Driver_Base (DEFERRED -- see
                          network_driver_base.py); not needed for eval-on-dumps.
    stage_eval          : meta-driven multi-metric reduce -> per-scene + ALL tables.
                          Reads per-segment dumps (the monolith's real dumps load via
                          dump_reader's qualname routing); decoupled from inference.

Divergences from monolith: the assembly item is a ``Single_Seq_Sample`` (attribute
access, not a dict); stage_list_segments' GT pull is joints-only (camera K/RT is on the
deferred GT surface); stage_vis is not ported (vis out of scope).
'''
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from hjlib_dataset_assembly.dataset_builder.dataset import Dataset_Single_Seq_Assembly

from hjlib_evaluation.eval_reducer import eval_dumps_against_gt
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.network_driver_base import Network_Driver_Base
from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import TestSet


def build_segment_tag(name_scene: str, name_seq: str, id_person: int, start: int, end: int) -> str:
    '''Filename tag for per-segment dumps: unique + sortable by scene/seq/time.'''
    return '%s__%s__p%02d__%07d_%07d' % (name_scene, name_seq, id_person, start, end)


def _tag_of_segment(seg: Test_Segment) -> str:
    return build_segment_tag(
        seg.name_scene, seg.name_seq, seg.id_person,
        seg.index_frame_original_start, seg.index_frame_original_end,
    )


def path_pkl_for_segment(path_dump_dir: str, seg: Test_Segment) -> str:
    return osp.join(path_dump_dir, '%s.pkl' % _tag_of_segment(seg))


class Tester:
    '''Holds the testing ingredients; stages consume them.

    - testset        : which segments to test (scene-level Test_Segment + seq-local divider)
    - assembly       : inference-input pipeline (built by build_test_assembly)
    - gt_provider    : world-space GT (required for stage_eval)
    - network_driver : concrete Network_Driver_Base; required only for stage_inference
    '''

    def __init__(
            self,
            testset:        TestSet,
            assembly:       Dataset_Single_Seq_Assembly,
            gt_provider:    Optional[GT_Provider_Base]    = None,
            network_driver: Optional[Network_Driver_Base] = None,
        ):
        self.testset        = testset
        self.assembly       = assembly
        self.gt_provider    = gt_provider
        self.network_driver = network_driver

    def stage_list_segments(
            self,
            print_every:   int  = 200,
            flag_pull_gt:  bool = False,
            path_dump_csv: Optional[str] = None,
        ) -> None:
        '''Traverse the TestSet + Assembly (+ optional GT-joints pull) for wiring
        sanity. Optional per-segment CSV dump. (GT pull is joints-only: camera K/RT is
        on the deferred GT surface.)'''
        print(self.testset.summary())
        print()

        csv_fp = None
        if path_dump_csv is not None:
            os.makedirs(osp.dirname(path_dump_csv), exist_ok=True)
            csv_fp = open(path_dump_csv, 'w')
            csv_fp.write('flat_idx,name_dataset,name_scene,name_seq,id_person,start_orig,end_orig,length\n')

        try:
            num_items = len(self.testset)
            for i in tqdm(range(num_items), desc='segments'):
                seg = self.testset.get_test_segment(i)
                item = self.assembly[i]

                if flag_pull_gt:
                    assert self.gt_provider is not None, (
                        'stage_list_segments(flag_pull_gt=True) requires a gt_provider.')
                    frame_range = (seg.index_frame_original_start, seg.index_frame_original_end)
                    joints_gt = self.gt_provider.get_smpl_joints_54_world(
                        seg.name_scene, seg.name_seq, frame_range)
                    assert joints_gt.shape[0] == seg.length, (joints_gt.shape, seg.length)

                if csv_fp is not None:
                    csv_fp.write('%d,%s,%s,%s,%d,%d,%d,%d\n' % (
                        i, seg.name_dataset, seg.name_scene, seg.name_seq, seg.id_person,
                        seg.index_frame_original_start, seg.index_frame_original_end, seg.length,
                    ))

                if (print_every > 0) and (i % print_every == 0 or i == num_items - 1):
                    kp2d_shape = tuple(item.keypoints_2D.shape)
                    tqdm.write('  [%4d/%d]  %s  item.kp2d.shape=%s'
                               % (i, num_items, seg.to_str(), kp2d_shape))
        finally:
            if csv_fp is not None:
                csv_fp.close()
                print('Wrote per-segment CSV: %s' % path_dump_csv)

    def stage_inference(
            self,
            path_dump_dir: str,
            case_filter: Optional[Set[str]] = None,
        ) -> None:
        '''Run network_driver.infer() per segment; dump one pkl per segment. Output
        filename = build_segment_tag(...) + ".pkl"; content =
        {'segment': Test_Segment, 'pred': <driver output dict>}. Requires a live
        Network_Driver_Base (deferred).'''
        assert self.network_driver is not None, (
            'stage_inference requires a Network_Driver_Base; none injected (the live '
            'driver is deferred -- see network_driver_base.py).')
        os.makedirs(path_dump_dir, exist_ok=True)
        num_items = len(self.testset)

        if case_filter is None:
            indices_to_run = list(range(num_items))
        else:
            tag_to_index: Dict[str, int] = {
                _tag_of_segment(self.testset.get_test_segment(i)): i for i in range(num_items)
            }
            missing = sorted(t for t in case_filter if t not in tag_to_index)
            assert not missing, (
                'stage_inference: case_filter contains %d tag(s) not in TestSet:\n  %s'
                % (len(missing), '\n  '.join(missing)))
            indices_to_run = sorted(tag_to_index[t] for t in case_filter)
            print('stage_inference: case_filter active -- running %d / %d segments'
                  % (len(indices_to_run), num_items))

        for i in tqdm(indices_to_run, desc='infer'):
            seg = self.testset.get_test_segment(i)
            item = self.assembly[i]
            pred = self.network_driver.infer({'sample': item})

            path_pkl = path_pkl_for_segment(path_dump_dir, seg)
            with open(path_pkl, 'wb') as f:
                pickle.dump({'segment': seg, 'pred': pred}, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Wrote %d per-segment pkls to %s' % (len(indices_to_run), path_dump_dir))

    def stage_eval(self, path_dump_dir: str, pred_joints_key: str = 'joints_54_world') -> None:
        '''Reduce per-segment pred dumps vs gt_provider GT into per-scene + ALL
        frame-weighted MPJPE / T-MPJPE tables, one per metric variant in the dataset's
        eval meta. NaN at meta-declared indices raises. Delegates to
        eval_reducer.eval_dumps_against_gt so an external-method tester reuses the exact
        same code path.

        pred_joints_key defaults to ``joints_54_world``, the monolith-equivalent
        dump-side tamed protocol. Use ``joints_54_world_raw`` only for a diagnostic
        no-invalid-tame pass when the dump carries that field.'''
        assert self.gt_provider is not None, (
            'stage_eval requires a gt_provider; none was injected.')
        eval_dumps_against_gt(
            testset              =self.testset,
            gt_provider          =self.gt_provider,
            path_dump_dir        =path_dump_dir,
            path_pkl_for_segment =path_pkl_for_segment,
            pred_joints_key      =pred_joints_key,
        )


def list_dump_segment_tags(path_dump_dir: str) -> List[str]:
    '''The segment tags present as ``<tag>.pkl`` in a dump dir (for restricting a
    TestSet to the segments a given dump actually covers -- monolith dumps may be a
    subset, e.g. a curated policy run).'''
    if not osp.isdir(path_dump_dir):
        return []
    return sorted(osp.splitext(n)[0] for n in os.listdir(path_dump_dir) if n.endswith('.pkl'))
