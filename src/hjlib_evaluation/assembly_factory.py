'''Factory: TestSet -> the inference-input assembly Dataset.

Port of monolith ``test/protocol_dynamic/assembly_factory.py``, rerouted to the
hjlib-dataset-assembly factory (DIV-1). The monolith hand-built
``Dataset_Single_Seq`` + ``Seq_Label_Manager`` + encoder + ``Dataset_Single_Seq_Assembly``
from ``local_setting`` paths; here ``get_dataset_seq_assembly`` encapsulates the
label-manager + Dataset wiring, and we inject ``testset.divider`` (built in lockstep
with ``testset.test_segments`` so the Dataset index == the GT-lookup index, and so
``TestSet.restrict_to_scenes`` + curated policies still propagate to the Dataset).

The encoder / image_reader / kp_manager are injection-style overrides (built by the
caller; the network's pre-encoded feature-cache encoder is wired in the
network_driver phase). With none, the assembly geometry-default ``matrix_transform``
gives the runnable label-only path.
'''
from __future__ import annotations

from typing import Any, Optional, Tuple

from hjlib_dataset_assembly.dataset_builder.dataset import (
    Dataset_Single_Seq_Assembly,
    get_dataset_seq_assembly,
)

from hjlib_evaluation.testset import TestSet


def build_test_assembly(
        testset:         TestSet,
        encoder:         Optional[Any] = None,
        image_reader:    Optional[Any] = None,
        kp_manager:      Optional[Any] = None,
        joint_format:    str = 'coco',
        dist_image_size: Tuple[int, int] = (256, 256),
        flag_test_mode:  bool = True,
    ) -> Dataset_Single_Seq_Assembly:
    '''Construct the test-time assembly Dataset backed by ``testset.divider``.

    @param encoder: ViT feature-cache encoder (supplies feature_token / the
        encoding-space transform); ``None`` -> geometry-default label-only path.
    @param kp_manager: detected-KP override (``build_kp_manager``); ``None`` = GT kps.
    @return: the ``Dataset_Single_Seq_Assembly`` (the monolith also returned a
        ``Dataset_Single_Seq`` disk manager; that is now internal to the factory).
    '''
    return get_dataset_seq_assembly(
        testset.path_root_label,
        testset.name_dataset,
        testset.fps,
        divider         =testset.divider,
        joint_format    =joint_format,
        dist_image_size =dist_image_size,
        flag_test_mode  =flag_test_mode,
        encoder         =encoder,
        image_reader    =image_reader,
        kp_manager      =kp_manager,
    )
