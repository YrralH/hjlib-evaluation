'''Reader for per-segment inference dumps (``{'segment': Test_Segment, 'pred': dict}``).

A dump pickles a ``Test_Segment`` (frozen dataclass) + an opaque prediction dict of
numpy arrays. The FROZEN monolith dumps pickle the monolith
``lib_dynamic_hvip...test_segment.Test_Segment`` qualname; importing it drags in the
monolith's heavy stack (network / yacs). So we route the qualname to OUR
``Test_Segment`` via a custom ``Unpickler.find_class`` (the hjlib-dataset-assembly
``legacy_pickle`` pattern) -- the read imports NO monolith code, numpy / builtins keep
their real reducers, and the new lib's OWN dumps (our qualname) load identically.

A frozen dataclass unpickles by ``cls.__new__`` + ``__dict__.update(state)`` (bypassing
``__setattr__``), so routing the monolith qualname to our same-field Test_Segment yields
a valid instance that compares equal by fields to a freshly-built one.

Legacy name normalization: the FROZEN monolith dumps pickled ``name_dataset`` as the
(now-deprecated) BARE dataset name (``worldpose`` / ``jta`` / ``jta_ext``). The assembly
registry has since made dataset names CANONICAL -- model + fit suffixed -- because the
bare names were variant-AMBIGUOUS (a bare ``rich`` could mean refit-SMPL or native-SMPL-X,
etc.). For the datasets the eval protocol covers the bare -> canonical variant is
unambiguous, so a legacy dump's bare ``name_dataset`` is mapped to the SPECIFIC canonical
variant it refers to (it is NOT dropped: ``name_dataset`` identifies the fit variant, which
the reducer's per-segment match still guards -- a dump for the wrong variant must fail).
The new lib's own future dumps already carry canonical names and pass through untouched.
'''
from __future__ import annotations

import dataclasses
import io
import pickle
from typing import Any, Dict, Tuple, cast, override

from hjlib_evaluation.test_segment import Test_Segment


# Monolith-era BARE dump name -> the specific canonical variant it denotes. Only the
# datasets the eval protocol evaluates (their bare -> canonical map is unambiguous; the
# variant-ambiguous bare names like ``rich`` are not in scope here). Mirrors the assembly
# registry deprecation intent without depending on its private table.
_LEGACY_DUMP_DATASET_TO_CANONICAL: Dict[str, str] = {
    'worldpose': 'worldpose_smpl',
    'jta':       'jta_smpl_fitted',
    'jta_ext':   'jta_ext_smpl_fitted',
}


class _Dump_Unpickler(pickle.Unpickler):
    '''Routes any ``Test_Segment`` qualname (monolith or hjlib) to our class;
    builtins / __builtin__ / collections / numpy* keep their real reducers (no monolith
    import). Any other class raises -- so a pred dict must be numpy-only (no torch).'''

    @override
    def find_class(self, module: str, name: str) -> Any:
        if module in ('builtins', '__builtin__', 'collections') or module.startswith('numpy'):
            return super().find_class(module, name)
        if 'Test_Segment' in name:
            return Test_Segment
        raise pickle.UnpicklingError('unexpected class in inference dump: %s.%s' % (module, name))


def load_inference_dump(path_pkl: str) -> Tuple[Test_Segment, Dict[str, Any]]:
    '''Read one per-segment dump -> (segment, pred_dict). The segment qualname is
    routed to our Test_Segment; the pred dict (plain numpy) is loaded as-is.'''
    with open(path_pkl, 'rb') as file:
        loaded = _Dump_Unpickler(io.BytesIO(file.read())).load()
    assert isinstance(loaded, dict), type(loaded)
    data = cast('Dict[str, Any]', loaded)
    segment = data['segment']
    assert isinstance(segment, Test_Segment), type(segment)
    canonical = _LEGACY_DUMP_DATASET_TO_CANONICAL.get(segment.name_dataset)
    if canonical is not None:
        # Legacy monolith dump: stamp the specific canonical variant (frozen -> replace).
        segment = dataclasses.replace(segment, name_dataset=canonical)
    pred = data['pred']
    assert isinstance(pred, dict), type(pred)
    return segment, cast('Dict[str, Any]', pred)
