'''Master smoke-test runner. Imports each per-topic smoke_test_* function and
runs them sequentially.

Why the __file__ + sys.path trick: test_smoke/ is intentionally not a package
(no __init__.py), so this script prepends its own directory to sys.path before
importing siblings by name when run as `python test_smoke/test_all_func.py`
from any cwd.
'''

import os.path as osp
import sys


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from test_testset import smoke_test_testset
from test_gt import smoke_test_gt
from test_ground_estimation_protocol import smoke_test_ground_estimation_protocol
from test_jta_person_detection import smoke_test_jta_person_detection
from test_jta_sota_metric_reducer import smoke_test_jta_sota_metric_reducer
from test_lsvhr_frame_visualization import smoke_test_lsvhr_frame_visualization
from test_lsvhr_evaluation import smoke_test_lsvhr_evaluation
from test_metric_leaves import smoke_test_metric_leaves
from test_eval_pred_field import smoke_test_eval_pred_field
from test_testset_fixed_window import smoke_test_testset_fixed_window
from test_trajectory_residual import smoke_test_trajectory_residual
from test_virtualcrowd_naive_comparison import smoke_test_virtualcrowd_naive_comparison
from test_corrected_crowd import smoke_test_corrected_crowd
from test_density_balanced_rcr_operation import smoke_test_density_balanced_rcr_operation
from test_density_balanced_rcr_cartesian import smoke_test_density_balanced_rcr_cartesian


def main() -> None:
    smoke_test_corrected_crowd()
    smoke_test_density_balanced_rcr_operation()
    smoke_test_density_balanced_rcr_cartesian()
    smoke_test_ground_estimation_protocol()
    smoke_test_jta_person_detection()
    smoke_test_jta_sota_metric_reducer()
    smoke_test_lsvhr_frame_visualization()
    smoke_test_lsvhr_evaluation()
    smoke_test_testset()
    smoke_test_gt()
    smoke_test_metric_leaves()
    smoke_test_eval_pred_field()
    smoke_test_testset_fixed_window()
    smoke_test_trajectory_residual()
    smoke_test_virtualcrowd_naive_comparison()
    print('test_all_func: all smoke tests passed')


if __name__ == '__main__':
    main()
