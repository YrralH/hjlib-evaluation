'''hjlib-evaluation: dynamic-scene evaluation protocol.

Execution layer symmetric to hjlib-trainer. It consumes a ready hjlib-network
estimator + a filter-configured hjlib-dataset-assembly Dataset, runs inference
over test segments, and reduces predictions against world-space GT into metric
tables. It owns no model and no dataset definition.

Top-level re-exports below cover the full ported surface (testset / GT / tester /
reducer / dump reader).
'''
from hjlib_evaluation.assembly_factory import build_test_assembly
from hjlib_evaluation.corrected_crowd_data import (
    CORRECTED_CROWD_METRICS,
    CORRECTED_CROWD_METRIC_UNITS,
    CORRECTED_CROWD_SCHEMA_VERSION,
    CORRECTED_CROWD_VIEWS,
    Corrected_Crowd_Result,
    Corrected_Crowd_Sequence,
    Corrected_Crowd_Sequence_Summary,
    corrected_crowd_result_to_json,
    corrected_crowd_summary_from_json,
    corrected_crowd_summary_to_json,
    validate_corrected_crowd_sequence,
)
from hjlib_evaluation.corrected_crowd_protocol import (
    evaluate_corrected_crowd_sequence,
    reduce_corrected_crowd_summaries,
)
from hjlib_evaluation.crowd_layout import (
    compute_pcod_3class_matches,
    compute_ppds_scores,
)
from hjlib_evaluation.dump_reader import load_inference_dump
from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_2D_OKS, Metric_Spec_3D
from hjlib_evaluation.eval_reducer import compute_jitter, eval_dumps_against_gt
from hjlib_evaluation.get_by_dataset import get_gt_provider, get_testset_builder
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.joint_error import compute_joint_position_errors
from hjlib_evaluation.joint_acceleration import compute_joint_acceleration_errors
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
    'CORRECTED_CROWD_METRICS',
    'CORRECTED_CROWD_METRIC_UNITS',
    'CORRECTED_CROWD_SCHEMA_VERSION',
    'CORRECTED_CROWD_VIEWS',
    'Corrected_Crowd_Result',
    'Corrected_Crowd_Sequence',
    'Corrected_Crowd_Sequence_Summary',
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
    'compute_joint_acceleration_errors',
    'compute_joint_position_errors',
    'compute_keypoint_oks_matrix',
    'compute_pcod_3class_matches',
    'compute_ppds_scores',
    'corrected_crowd_result_to_json',
    'corrected_crowd_summary_from_json',
    'corrected_crowd_summary_to_json',
    'eval_dumps_against_gt',
    'evaluate_corrected_crowd_sequence',
    'get_gt_provider',
    'get_testset_builder',
    'list_dump_segment_tags',
    'load_inference_dump',
    'path_pkl_for_segment',
    'reduce_trajectory_residual_summaries',
    'reduce_corrected_crowd_summaries',
    'summarize_trajectory_residuals',
    'validate_corrected_crowd_sequence',
]
