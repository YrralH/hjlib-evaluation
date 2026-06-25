'''Abstract network driver: wraps 'load model + run inference on one item'.

Port of monolith ``test/protocol_dynamic/network_driver_base.py``. The Tester's
stage_inference calls ``driver.infer(dict_item)`` per Test_Segment and does not know
about checkpoint paths, exp config, or model classes. Concrete drivers (a live
hjlib-network ``Seq_Estimator`` adapter -- load ckpt + translate the assembly
``Single_Seq_Sample_Batch`` into the network's ``dict_batch``, mirroring the
hjlib-experiments ``adapt_batch_to_dict_batch``) live outside this lib so this module
stays model-free.

Live-driver status (2026-06-24): DEFERRED. The eval-protocol migration + parity run
on the monolith's existing per-segment prediction dumps (read via ``dump_reader``),
which decouple eval from inference -- so the metric baseline needs no live network.
A concrete live driver is only needed to score a NEW ckpt; it is a later increment
(see docs/design/migration.md / EVAL-DICTBATCH Cross-lib TODO).
'''
from __future__ import annotations

import abc
from typing import Any, Dict


class Network_Driver_Base(abc.ABC):

    @abc.abstractmethod
    def infer(self, dict_item: Dict[str, Any]) -> Dict[str, Any]:
        '''Run inference on one segment's input and return a prediction dict.

        Input: stage_inference passes ``{'sample': item}`` where ``item`` is this
        Test_Segment's assembly item (a ``Single_Seq_Sample``, attribute-access -- NOT a
        plain dict). Output contract: ``pred['joints_54_world']`` of shape ``(L, 54, 3)``
        (what eval_reducer reads). stage_inference treats ``pred`` as opaque and only
        pickles it, so it MUST be numpy-only -- the ``dump_reader`` whitelist rejects
        torch tensors, so a live driver converts tensors to numpy before returning.
        (live driver deferred, DIV-10; the wrapper shape + this contract get finalized
        when it lands.)'''
        raise NotImplementedError
