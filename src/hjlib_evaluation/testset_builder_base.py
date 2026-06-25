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
    '''One configured instance per dataset. Reads the dataset's ``Seq_Modification``
    lists from the washed hjlib-dataset-assembly filter store and returns a unified
    TestSet (divider + scene-level Test_Segment list, index-aligned).'''

    name_dataset: str = ''

    @abc.abstractmethod
    def build(self, policy: str, split: str) -> TestSet:
        '''
        @param policy: filter policy. ``full`` (test-set eval policy) or ``visualize``
            (its superset). Curated subsets (``visualize_v2`` / ``visualize_v3`` /
            ``*_smoke``) are not yet supported -- see EVAL Cross-lib TODO / migration.md.
        @param split: ``train`` / ``val`` / ``test`` / ``all`` (read from the dump
            root's split txt; ``all`` = the sorted union).
        '''
        raise NotImplementedError
