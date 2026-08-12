'''hjlib-evaluation: dynamic-scene evaluation protocol.

Execution layer symmetric to hjlib-trainer. It consumes a ready hjlib-network
estimator + a filter-configured hjlib-dataset-assembly Dataset, runs inference
over test segments, and reduces predictions against world-space GT into metric
tables. It owns no model and no dataset definition.

Top-level re-exports below cover the full ported surface (testset / GT / tester /
reducer / dump reader).
'''
from hjlib_evaluation.assembly_factory import build_test_assembly
from hjlib_evaluation.dump_reader import load_inference_dump
from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_2D_OKS, Metric_Spec_3D
from hjlib_evaluation.eval_reducer import compute_jitter, eval_dumps_against_gt
from hjlib_evaluation.get_by_dataset import get_gt_provider, get_testset_builder
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.joint_error import compute_joint_position_errors
from hjlib_evaluation.keypoint_oks import compute_keypoint_oks_matrix
from hjlib_evaluation.network_driver_base import Network_Driver_Base
from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import Filter_Stats, TestSet
from hjlib_evaluation.testset_builder import TestSet_Builder
from hjlib_evaluation.testset_builder_base import TestSet_Builder_Base
from hjlib_evaluation.tester import Tester, build_segment_tag, list_dump_segment_tags, path_pkl_for_segment
from hjlib_evaluation.trajectory_residual import (
    Trajectory_Residual_Reduction,
    Trajectory_Residual_Summary,
    reduce_trajectory_residual_summaries,
    summarize_trajectory_residuals,
)

__all__ = [
    'Eval_Meta',
    'Filter_Stats',
    'GT_Provider_Base',
    'Metric_Spec_2D_OKS',
    'Metric_Spec_3D',
    'Network_Driver_Base',
    'TestSet',
    'TestSet_Builder',
    'TestSet_Builder_Base',
    'Test_Segment',
    'Tester',
    'Trajectory_Residual_Reduction',
    'Trajectory_Residual_Summary',
    'build_segment_tag',
    'build_test_assembly',
    'compute_jitter',
    'compute_joint_position_errors',
    'compute_keypoint_oks_matrix',
    'eval_dumps_against_gt',
    'get_gt_provider',
    'get_testset_builder',
    'list_dump_segment_tags',
    'load_inference_dump',
    'path_pkl_for_segment',
    'reduce_trajectory_residual_summaries',
    'summarize_trajectory_residuals',
]
