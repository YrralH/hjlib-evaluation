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
from hjlib_evaluation.network_driver_base import Network_Driver_Base
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


class Tiny_Assembly:
    def __init__(self, testset: Eval_TestSet) -> None:
        self.divider = testset.divider
        self.name_dataset = testset.name_dataset

    def __len__(self) -> int:
        return len(self.divider)

    def __getitem__(self, index: int) -> object:
        return {'index': index}


class Tiny_Driver(Network_Driver_Base):
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    @override
    def infer(self, dict_item: Dict[str, Any]) -> Dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {'item': dict_item['sample']}


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
        name_seq='0000_0000',
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


def test_eval_rejects_empty_testset_and_stale_segment(tmp_path: Path) -> None:
    empty = Eval_TestSet(
        name_dataset='worldpose_smpl',
        policy='tiny',
        split='test',
        divider=Filtered_Sub_Seq_Divider([]),
        test_segments=[],
        path_root_label='unused',
        fps=30.0,
    )
    try:
        eval_dumps_against_gt(
            empty, Tiny_GT_Provider(), str(tmp_path), path_pkl_for_segment)
    except ValueError as error:
        assert 'nonempty TestSet' in str(error)
    else:
        raise AssertionError('empty evaluation must fail')

    seg = Eval_Test_Segment(
        'worldpose_smpl', 'scene', '0000_0000', 0, 0, 2)
    stale = Eval_Test_Segment(
        'worldpose_smpl', 'other', '0000_0000', 0, 0, 2)
    path = path_pkl_for_segment(str(tmp_path), seg)
    with open(path, 'wb') as file:
        pickle.dump(
            {'segment': stale, 'pred': {'joints_54_world': np.zeros((2, 54, 3))}},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    try:
        eval_dumps_against_gt(
            make_tiny_testset(seg), Tiny_GT_Provider(), str(tmp_path),
            path_pkl_for_segment)
    except ValueError as error:
        assert 'different segment' in str(error)
    else:
        raise AssertionError('stale segment identity must fail')


def test_inference_publishes_once_and_cleans_failure(tmp_path: Path) -> None:
    seg = Eval_Test_Segment(
        'worldpose_smpl', 'scene', '0000_0000', 0, 0, 2)
    testset = make_tiny_testset(seg)
    assembly = cast(Any, Tiny_Assembly(testset))
    output = tmp_path / 'result'
    Eval_Tester(
        testset,
        assembly,
        network_driver=Tiny_Driver(),
    ).stage_inference(str(output))
    assert Path(path_pkl_for_segment(str(output), seg)).is_file()
    try:
        Eval_Tester(
            testset,
            assembly,
            network_driver=Tiny_Driver(),
        ).stage_inference(str(output))
    except FileExistsError:
        pass
    else:
        raise AssertionError('existing inference output must not be overwritten')

    failed_output = tmp_path / 'failed'
    expected_error = RuntimeError('inference failed')
    try:
        Eval_Tester(
            testset,
            assembly,
            network_driver=Tiny_Driver(expected_error),
        ).stage_inference(str(failed_output))
    except RuntimeError as error:
        assert error is expected_error
    else:
        raise AssertionError('driver failure must propagate')
    assert not failed_output.exists()
    assert not tuple(tmp_path.glob('.failed.partial-*'))

    wrong_assembly = Tiny_Assembly(testset)
    wrong_assembly.divider = Filtered_Sub_Seq_Divider([
        ('scene', '0000_0000', 0, 2),
    ])
    try:
        Eval_Tester(
            testset,
            cast(Any, wrong_assembly),
            network_driver=Tiny_Driver(),
        ).stage_inference(str(tmp_path / 'wrong'))
    except ValueError as error:
        assert 'identity-aligned' in str(error)
    else:
        raise AssertionError('equal-but-distinct divider must fail binding')


def smoke_test_eval_pred_field() -> None:
    with TemporaryDirectory() as tmp:
        test_eval_dumps_can_select_tamed_or_raw_prediction_field(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_eval_rejects_empty_testset_and_stale_segment(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_inference_publishes_once_and_cleans_failure(Path(tmp))
