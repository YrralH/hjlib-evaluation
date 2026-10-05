'''Per-dataset factory: name_dataset -> a configured TestSet_Builder / GT_Provider.

Port of monolith ``test/protocol_dynamic/get_by_dataset.py``. The monolith returned
bare per-dataset classes that read paths from ``local_setting``; hjlib injects the
data roots (family rule), resolves the per-dataset facts (dump folder + fps via the
assembly registry) + the filter-store token, and returns a ready-to-use builder /
provider.

GT-provider data sources: WP and VRv1 world-space SMPL come
from the assembly dump (full label) -> need only the dump root; JTA / JTA_Ext raw
22-joint GT comes from hjlib-dataset-std (raw data root). Camera / ground / video GT
(2D OKS + vis) is deferred (base raises).

Per-dataset token quirks preserved from the monolith:
- the dump-label FOLDER is the assembly leaf-suffixed name (``jta_smpl_fitted`` ...,
  plain ``worldpose``), resolved via ``get_dataset_facts``;
- the filter-stats DIR token is the plain name EXCEPT worldpose, whose monolith dir
  is ``wp_filter_stats`` (not ``worldpose_filter_stats``).
'''
from __future__ import annotations

import os.path as osp
from typing import Dict, Optional

from hjlib_dataset_assembly.dataset_builder.dataset_registry import get_dataset_facts
from hjlib_dataset_std import Joints_Only_Dynamic_Std, get_jta_ext_std, get_jta_std

from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.dumped_smpl_gt_provider import Dumped_SMPL_GT_Provider
from hjlib_evaluation.per_dataset.gt_provider_jta import JTA_Ext_GT_Provider, JTA_GT_Provider
from hjlib_evaluation.per_dataset.gt_provider_wp import WP_GT_Provider
from hjlib_evaluation.per_dataset.testset_builder_vrv1 import VRv1_TestSet_Builder
from hjlib_evaluation.per_dataset.vrv1_eval_meta import (
    VRV1_EVAL_META, VRV1_GOPRO_EVAL_META,
)
from hjlib_evaluation.testset_builder import TestSet_Builder
from hjlib_evaluation.testset_builder_base import TestSet_Builder_Base


# Canonical names. VRv1 consumes whole dumps rather than a washed filter store.
DATASETS_SUPPORTED = (
    'worldpose_smpl', 'jta_smpl_fitted', 'jta_ext_smpl_fitted',
    'vrv1_smpl_fitted', 'vrv1_gopro_smpl_fitted',
)
_VRV1_META = {
    'vrv1_smpl_fitted': VRV1_EVAL_META,
    'vrv1_gopro_smpl_fitted': VRV1_GOPRO_EVAL_META,
}

# filter-stats dir token: the on-disk ``<token>_filter_stats/`` dirs keep the PLAIN
# raw-dataset token (``wp`` for worldpose -- monolith quirk -- and the bare ``jta`` /
# ``jta_ext`` for the others), so each canonical name maps to its plain dir token.
_FILTER_STATS_TOKEN: Dict[str, str] = {
    'worldpose_smpl':      'wp',
    'jta_smpl_fitted':     'jta',
    'jta_ext_smpl_fitted': 'jta_ext',
}

_SUFFIX_FILTER_STORE = 'seq_modifications_jsonbin'


def get_testset_builder(
        name_dataset: str,
        path_dump_root: str,
        path_filter_stats_base: str = '',
        filter_version: Optional[str] = None,
    ) -> TestSet_Builder_Base:
    '''Build the dataset's filtered or full-dump TestSet builder.

    @param path_dump_root: the dump root holding ``<dataset>_<leaf>/`` folders
        (monolith ``__As_Single_Bbox_hjlib__``).
    @param path_filter_stats_base: the base holding ``<token>_filter_stats/`` dirs;
        not needed for VRv1's full-policy dumped-run builder.
    @param filter_version: filter-store version (``None`` follows current).
    '''
    if name_dataset == 'vrv1':
        get_dataset_facts(name_dataset)  # Registry owns deprecated-name guidance.
    assert name_dataset in DATASETS_SUPPORTED, (
        'no TestSet_Builder for %r; supported: %s' % (name_dataset, list(DATASETS_SUPPORTED)))

    facts = get_dataset_facts(name_dataset)
    path_root_label = osp.join(path_dump_root, facts.folder)
    if name_dataset in _VRV1_META:
        return VRv1_TestSet_Builder(name_dataset, path_root_label, facts.fps)
    token = _FILTER_STATS_TOKEN.get(name_dataset, name_dataset)
    path_filter_store = osp.join(path_filter_stats_base, '%s_filter_stats' % token, _SUFFIX_FILTER_STORE)

    return TestSet_Builder(
        name_dataset      =name_dataset,
        path_root_label   =path_root_label,
        path_filter_store =path_filter_store,
        fps               =facts.fps,
        filter_version    =filter_version,
    )


def get_gt_provider(
        name_dataset: str,
        path_dump_root: str,
        path_raw_data_root: Optional[str] = None,
        path_raw_more_label: Optional[str] = None,
    ) -> GT_Provider_Base:
    '''Build a configured GT_Provider for a dataset (metric-baseline scope).

    @param path_dump_root: dump root holding ``<dataset>_<leaf>/`` (WP/VRv1 GT source).
    @param path_raw_data_root: the dataset's RAW data root -- required for jta /
        jta_ext (their GT joints come from hjlib-dataset-std), unused for WP/VRv1.
    @param path_raw_more_label: optional raw more-label root (jta / jta_ext).
    '''
    if name_dataset == 'vrv1':
        get_dataset_facts(name_dataset)  # Registry owns deprecated-name guidance.
    assert name_dataset in DATASETS_SUPPORTED, (
        'no GT_Provider for %r; supported: %s' % (name_dataset, list(DATASETS_SUPPORTED)))

    if name_dataset in _VRV1_META:
        facts = get_dataset_facts(name_dataset)
        return Dumped_SMPL_GT_Provider(
            name_dataset, osp.join(path_dump_root, facts.folder), _VRV1_META[name_dataset])

    if name_dataset == 'worldpose_smpl':
        facts = get_dataset_facts('worldpose_smpl')
        return WP_GT_Provider(path_root_label=osp.join(path_dump_root, facts.folder))

    # jta / jta_ext: raw 22-joint GT via hjlib-dataset-std (label='joints').
    assert path_raw_data_root is not None, (
        '%s GT needs path_raw_data_root (the raw dataset root)' % name_dataset)
    if name_dataset == 'jta_smpl_fitted':
        std = get_jta_std(path_raw_data_root, 'joints', path_more_label=path_raw_more_label)
        assert isinstance(std, Joints_Only_Dynamic_Std), type(std)
        return JTA_GT_Provider(std)
    std = get_jta_ext_std(path_raw_data_root, 'joints', path_more_label=path_raw_more_label)
    assert isinstance(std, Joints_Only_Dynamic_Std), type(std)
    return JTA_Ext_GT_Provider(std)
