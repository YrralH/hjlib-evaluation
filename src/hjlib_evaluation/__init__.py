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
    CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
    CORRECTED_CROWD_SCHEMA_VERSION,
    CORRECTED_CROWD_VIEWS,
    Corrected_Crowd_Result,
    Corrected_Crowd_Selected_View_Result,
    Corrected_Crowd_Selected_View_Sequence_Summary,
    Corrected_Crowd_Sequence,
    Corrected_Crowd_Sequence_Summary,
    corrected_crowd_result_to_json,
    corrected_crowd_selected_view_result_from_json,
    corrected_crowd_selected_view_result_to_json,
    corrected_crowd_selected_view_summary_from_json,
    corrected_crowd_selected_view_summary_to_json,
    corrected_crowd_summary_from_json,
    corrected_crowd_summary_to_json,
    validate_corrected_crowd_sequence,
    validate_corrected_crowd_selected_view_name,
)
from hjlib_evaluation.corrected_crowd_population import (
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
    make_coco17_visible_ge9_common_mask,
)
from hjlib_evaluation.corrected_crowd_protocol import (
    evaluate_corrected_crowd_selected_view,
    evaluate_corrected_crowd_sequence,
    reduce_corrected_crowd_selected_view_summaries,
    reduce_corrected_crowd_summaries,
)
from hjlib_evaluation.corrected_crowd_world_dynamics import (
    CORRECTED_CROWD_WORLD_DYNAMICS_METRICS,
    CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION,
    CORRECTED_CROWD_WORLD_DYNAMICS_UNITS,
    Corrected_Crowd_World_Dynamics_Result,
    Corrected_Crowd_World_Dynamics_Sequence_Summary,
    corrected_crowd_world_dynamics_result_from_json,
    corrected_crowd_world_dynamics_result_to_json,
    corrected_crowd_world_dynamics_summary_from_json,
    corrected_crowd_world_dynamics_summary_to_json,
    evaluate_corrected_crowd_selected_view_and_world_dynamics,
    evaluate_corrected_crowd_world_dynamics,
    reduce_corrected_crowd_world_dynamics_summaries,
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
from hjlib_evaluation.ground_estimation_protocol import (
    Ground_Effect_Decomposition,
    Ground_Effect_Support,
    Ground_Estimation_Result,
    Ground_Estimator,
    Ground_Observation_Set,
    Ground_Plane_Diagnostics,
    collect_ground_observations,
    compute_ground_effect_decomposition,
    compute_ground_plane_diagnostics,
    compute_same_ray_ground_errors,
    estimate_ground_from_observations,
    lower_weighted_median,
    sample_ground_observations,
    select_ground_observations_at_frame,
    summarize_ground_errors,
    take_ground_observations,
    validate_ground_effect_support_against_K,
)
from hjlib_evaluation.joint_error import compute_joint_position_errors
from hjlib_evaluation.joint_acceleration import compute_joint_acceleration_errors
from hjlib_evaluation.joint_jerk import compute_joint_jerk_errors
from hjlib_evaluation.jta_person_detection_data import (
    JTA_CAMERA_K,
    JTA_ENDPOINT_INDICES,
    JTA_ENDPOINT_NAMES,
    JTA_ENDPOINT_OKS_SIGMAS,
    JTA_IMAGE_SIZE_WH,
    JTA_PERSON_DETECTION_SCHEMA_VERSION,
    SMPL54_ENDPOINT_INDICES,
    JTA_Person_Detection_GT_Frame,
    JTA_Person_Detection_Prediction_Frame,
    JTA_Person_Detection_Result,
    jta_person_detection_result_from_json,
    jta_person_detection_result_to_json,
    make_jta_person_detection_gt_frame,
)
from hjlib_evaluation.jta_person_detection_protocol import (
    JTA_Person_Association,
    JTA_Person_Detection_Frame_Metrics,
    JTA_Person_Detection_Reducer,
    OKS_QUANTIZATION,
    OKS_THRESHOLD,
    associate_jta_people,
    evaluate_jta_person_detection_frame,
)
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
    'CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION',
    'CORRECTED_CROWD_SCHEMA_VERSION',
    'CORRECTED_CROWD_VIEWS',
    'CORRECTED_CROWD_WORLD_DYNAMICS_METRICS',
    'CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION',
    'CORRECTED_CROWD_WORLD_DYNAMICS_UNITS',
    'C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9',
    'Corrected_Crowd_Result',
    'Corrected_Crowd_Selected_View_Result',
    'Corrected_Crowd_Selected_View_Sequence_Summary',
    'Corrected_Crowd_Sequence',
    'Corrected_Crowd_Sequence_Summary',
    'Corrected_Crowd_World_Dynamics_Result',
    'Corrected_Crowd_World_Dynamics_Sequence_Summary',
    'Eval_Meta',
    'Filter_Stats',
    'GT_Provider_Base',
    'Ground_Effect_Decomposition',
    'Ground_Effect_Support',
    'Ground_Estimation_Result',
    'Ground_Estimator',
    'Ground_Observation_Set',
    'Ground_Plane_Diagnostics',
    'JTA_CAMERA_K',
    'JTA_ENDPOINT_INDICES',
    'JTA_ENDPOINT_NAMES',
    'JTA_ENDPOINT_OKS_SIGMAS',
    'JTA_IMAGE_SIZE_WH',
    'JTA_PERSON_DETECTION_SCHEMA_VERSION',
    'JTA_Person_Association',
    'JTA_Person_Detection_Frame_Metrics',
    'JTA_Person_Detection_GT_Frame',
    'JTA_Person_Detection_Prediction_Frame',
    'JTA_Person_Detection_Reducer',
    'JTA_Person_Detection_Result',
    'Metric_Spec_2D_OKS',
    'Metric_Spec_3D',
    'Network_Driver_Base',
    'OKS_QUANTIZATION',
    'OKS_THRESHOLD',
    'SMPL54_ENDPOINT_INDICES',
    'TestSet',
    'TestSet_Builder',
    'TestSet_Builder_Base',
    'Test_Segment',
    'Tester',
    'Trajectory_Residual_Reduction',
    'Trajectory_Residual_Summary',
    'build_segment_tag',
    'build_test_assembly',
    'associate_jta_people',
    'compute_jitter',
    'compute_joint_acceleration_errors',
    'compute_joint_jerk_errors',
    'compute_joint_position_errors',
    'collect_ground_observations',
    'compute_ground_effect_decomposition',
    'compute_ground_plane_diagnostics',
    'compute_same_ray_ground_errors',
    'compute_keypoint_oks_matrix',
    'compute_pcod_3class_matches',
    'compute_ppds_scores',
    'corrected_crowd_result_to_json',
    'corrected_crowd_selected_view_result_from_json',
    'corrected_crowd_selected_view_result_to_json',
    'corrected_crowd_selected_view_summary_from_json',
    'corrected_crowd_selected_view_summary_to_json',
    'corrected_crowd_summary_from_json',
    'corrected_crowd_summary_to_json',
    'corrected_crowd_world_dynamics_result_from_json',
    'corrected_crowd_world_dynamics_result_to_json',
    'corrected_crowd_world_dynamics_summary_from_json',
    'corrected_crowd_world_dynamics_summary_to_json',
    'eval_dumps_against_gt',
    'evaluate_corrected_crowd_sequence',
    'evaluate_corrected_crowd_selected_view',
    'evaluate_corrected_crowd_selected_view_and_world_dynamics',
    'evaluate_corrected_crowd_world_dynamics',
    'evaluate_jta_person_detection_frame',
    'estimate_ground_from_observations',
    'get_gt_provider',
    'get_testset_builder',
    'list_dump_segment_tags',
    'lower_weighted_median',
    'load_inference_dump',
    'jta_person_detection_result_from_json',
    'jta_person_detection_result_to_json',
    'make_jta_person_detection_gt_frame',
    'path_pkl_for_segment',
    'reduce_trajectory_residual_summaries',
    'reduce_corrected_crowd_summaries',
    'reduce_corrected_crowd_selected_view_summaries',
    'reduce_corrected_crowd_world_dynamics_summaries',
    'make_coco17_visible_ge9_common_mask',
    'sample_ground_observations',
    'select_ground_observations_at_frame',
    'summarize_ground_errors',
    'summarize_trajectory_residuals',
    'take_ground_observations',
    'validate_corrected_crowd_sequence',
    'validate_corrected_crowd_selected_view_name',
    'validate_ground_effect_support_against_K',
]
