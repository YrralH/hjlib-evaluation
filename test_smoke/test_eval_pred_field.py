from __future__ import annotations

import pickle
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, cast, override

import numpy as np

from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider

from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_3D
from hjlib_evaluation.eval_reducer import eval_dumps_against_gt
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.test_segment import Test_Segment as Eval_Test_Segment
from hjlib_evaluation.tester import Tester as Eval_Tester, path_pkl_for_segment
from hjlib_evaluation.testset import TestSet as Eval_TestSet


class Tiny_GT_Provider(GT_Provider_Base):
    name_dataset = 'worldpose_smpl'

    @override
    def get_smpl_joints_54_world(
            self,
            name_scene: str,
            name_seq: str,
            frame_range_scene_level: Tuple[int, int],
        ) -> np.ndarray:
        del name_scene, name_seq
        start, end = frame_range_scene_level
        return np.zeros((end - start, 54, 3), dtype=np.float32)

    @override
    def get_smpl_param_world(
            self,
            name_scene: str,
            name_seq: str,
            frame_range_scene_level: Tuple[int, int],
        ) -> Dict[str, np.ndarray]:
        del name_scene, name_seq, frame_range_scene_level
        raise NotImplementedError

    @override
    def get_eval_meta(self) -> Eval_Meta:
        return Eval_Meta(
            name_dataset='worldpose_smpl',
            meta_version='tiny',
            unit_world='m',
            k_rt_relation='shared',
            metrics_3d=(
                Metric_Spec_3D(
                    name='tiny',
                    joint_indices_smpl_54=(0, 1),
                    root_indices_smpl_54_for_alignment=(0,),
                ),
            ),
            metrics_2d_oks=(),
        )


def make_tiny_testset(seg: Eval_Test_Segment) -> Eval_TestSet:
    return Eval_TestSet(
        name_dataset='worldpose_smpl',
        policy='tiny',
        split='test',
        divider=Filtered_Sub_Seq_Divider([
            (seg.name_scene, seg.name_seq, 0, seg.length),
        ]),
        test_segments=[seg],
        path_root_label='unused',
        fps=30.0,
    )


def parse_all_mpjpe_tmpjpe(output: str) -> Tuple[float, float]:
    for line in output.splitlines():
        if line.startswith('ALL'):
            parts = line.split()
            return float(parts[1]), float(parts[2])
    raise AssertionError(output)


def test_eval_dumps_can_select_tamed_or_raw_prediction_field(tmp_path: Path) -> None:
    seg = Eval_Test_Segment(
        name_dataset='worldpose_smpl',
        name_scene='scene',
        name_seq='seq',
        id_person=0,
        index_frame_original_start=0,
        index_frame_original_end=2,
    )
    pred_tame = np.zeros((2, 54, 3), dtype=np.float32)
    pred_raw = np.zeros((2, 54, 3), dtype=np.float32)
    pred_raw[:, 0, 0] = 1.0
    pred_raw[:, 1, 0] = 3.0

    path = path_pkl_for_segment(str(tmp_path), seg)
    with open(path, 'wb') as file:
        pickle.dump({
            'segment': seg,
            'pred': {
                'joints_54_world': pred_tame,
                'joints_54_world_raw': pred_raw,
            },
        }, file, protocol=pickle.HIGHEST_PROTOCOL)

    out_tame = StringIO()
    with redirect_stdout(out_tame):
        eval_dumps_against_gt(make_tiny_testset(seg), Tiny_GT_Provider(), str(tmp_path), path_pkl_for_segment)
    out_raw = StringIO()
    with redirect_stdout(out_raw):
        Eval_Tester(
            make_tiny_testset(seg), cast(Any, object()), gt_provider=Tiny_GT_Provider(),
        ).stage_eval(
            str(tmp_path),
            pred_joints_key='joints_54_world_raw')

    assert parse_all_mpjpe_tmpjpe(out_tame.getvalue()) == (0.0, 0.0)
    assert parse_all_mpjpe_tmpjpe(out_raw.getvalue()) == (2000.0, 1000.0)

    try:
        eval_dumps_against_gt(
            make_tiny_testset(seg), Tiny_GT_Provider(), str(tmp_path), path_pkl_for_segment,
            pred_joints_key='missing_field')
    except KeyError as exc:
        assert 'missing_field' in str(exc)
    else:
        raise AssertionError('missing pred_joints_key should raise KeyError')


def smoke_test_eval_pred_field() -> None:
    with TemporaryDirectory() as tmp:
        test_eval_dumps_can_select_tamed_or_raw_prediction_field(Path(tmp))
