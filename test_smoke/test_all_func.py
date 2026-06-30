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
from test_eval_pred_field import smoke_test_eval_pred_field


def main() -> None:
    smoke_test_testset()
    smoke_test_gt()
    smoke_test_eval_pred_field()
    print('test_all_func: all smoke tests passed')


if __name__ == '__main__':
    main()
