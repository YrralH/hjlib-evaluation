'''Smoke: the GT / eval-meta layer's data-free units. No external data.

Covers the Eval_Meta contract (WP full-SMPL-24 + JTA limb-endpoint metrics) and
get_gt_provider error paths. The provider joint extraction (WP from assembly dump,
JTA from dataset-std raw) needs real data -> validated in test/.
'''
import os.path as osp
import sys

from hjlib_evaluation import Eval_Meta, get_gt_provider
from hjlib_evaluation.per_dataset.jta_eval_meta import JTA_EVAL_META, JTA_EXT_EVAL_META
from hjlib_evaluation.per_dataset.wp_eval_meta import WP_EVAL_META


sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


def _check_eval_meta() -> None:
    assert isinstance(WP_EVAL_META, Eval_Meta)
    assert WP_EVAL_META.name_dataset == 'worldpose_smpl'
    assert len(WP_EVAL_META.metrics_3d) == 1
    assert WP_EVAL_META.metrics_3d[0].joint_indices_smpl_54 == tuple(range(24))
    assert len(WP_EVAL_META.metrics_3d[0].root_indices_smpl_54_for_alignment) == 1

    for meta, nd in ((JTA_EVAL_META, 'jta_smpl_fitted'), (JTA_EXT_EVAL_META, 'jta_ext_smpl_fitted')):
        assert meta.name_dataset == nd, meta.name_dataset
        assert len(meta.metrics_3d) == 2, len(meta.metrics_3d)
        by_name = {m.name: m for m in meta.metrics_3d}
        assert len(by_name['limb_endpoints_with_hip'].joint_indices_smpl_54) == 12
        assert len(by_name['limb_endpoints_no_hip'].joint_indices_smpl_54) == 10
        # alignment root = hip pair in both
        for m in meta.metrics_3d:
            assert len(m.root_indices_smpl_54_for_alignment) == 2, m.name


def _check_get_gt_provider_errors() -> None:
    try:
        get_gt_provider('vrv1', path_dump_root='/dump')
    except NotImplementedError:
        pass
    else:
        raise AssertionError('vrv1 GT should raise NotImplementedError')

    try:
        get_gt_provider('nope', path_dump_root='/dump')
    except AssertionError:
        pass
    else:
        raise AssertionError('unknown dataset GT should assert')

    # jta GT needs a raw data root
    try:
        get_gt_provider('jta_smpl_fitted', path_dump_root='/dump', path_raw_data_root=None)
    except AssertionError:
        pass
    else:
        raise AssertionError('jta GT without path_raw_data_root should assert')


def test_eval_meta() -> None:
    _check_eval_meta()


def test_get_gt_provider_errors() -> None:
    _check_get_gt_provider_errors()


def smoke_test_gt() -> None:
    _check_eval_meta()
    _check_get_gt_provider_errors()
    print('smoke_test_gt: all checks passed')


if __name__ == '__main__':
    smoke_test_gt()
