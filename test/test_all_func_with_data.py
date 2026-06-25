'''Master runner for the data-dependent test/ tree (real washed store + dumped labels
+ raw dataset roots + monolith inference dumps). FAIL-not-skip per family policy.

Run: python test/test_all_func_with_data.py   (or pytest test/ -q)
'''
import os.path as osp
import sys


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from test_testset_with_data import smoke_test_testset_with_data
from test_eval_on_dumps_with_data import smoke_test_eval_on_dumps_with_data


def main() -> None:
    smoke_test_testset_with_data()
    smoke_test_eval_on_dumps_with_data()
    print('test_all_func_with_data: all data-dependent tests passed')


if __name__ == '__main__':
    main()
