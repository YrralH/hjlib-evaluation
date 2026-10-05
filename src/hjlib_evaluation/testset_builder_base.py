'''Abstract per-dataset builder that produces a TestSet.

Port of monolith ``test/protocol_dynamic/testset_builder_base.py``. The monolith
resolved the filter pkl dir + single-seq root from ``local_setting``; hjlib injects
those roots at construction (family rule: lib code holds no absolute data paths),
so the concrete builder takes them in ``__init__`` and ``build`` keeps the
monolith ``(policy, split)`` signature.
'''
from __future__ import annotations

import abc

from hjlib_evaluation.testset import TestSet


class TestSet_Builder_Base(abc.ABC):
    '''One configured instance per dataset, returning a unified TestSet.

    The divider and scene-level Test_Segment list are index-aligned. Concrete
    builders select washed filter-store ranges or complete frozen dump runs.
    '''

    name_dataset: str = ''
    path_root_label: str
    fps: float

    @abc.abstractmethod
    def build(self, policy: str, split: str) -> TestSet:
        '''
        @param policy: dataset-specific evaluation policy. Filtered builders
            accept ``full`` or ``visualize``; VRv1 dumped runs accept only ``full``.
        @param split: ``train`` / ``val`` / ``test`` / ``all`` (read from the dump
            root's split txt; ``all`` = the sorted union).
        '''
        raise NotImplementedError
