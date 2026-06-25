'''Test_Segment: per-item metadata shared across datasets.

Minimal: identifies 'which segment of which raw seq of which scene of which
dataset'. Does NOT carry K/RT (method K/RT rides with the prediction, GT K/RT
comes from the GT_Provider) and does NOT carry model input (that lives in the
assembly item).

Port of monolith ``test/protocol_dynamic/test_segment.py`` (verbatim).
'''
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Test_Segment:
    '''One segment in a TestSet. Scene-level frame coords (absolute in the raw
    dataset video / scene), so GT lookup just passes (name_scene, name_seq,
    frame_range) without any further offset math.'''
    name_dataset:               str
    name_scene:                 str
    name_seq:                   str       # raw-dataset sequence id (e.g. WP '0003_0000')
    id_person:                  int
    index_frame_original_start: int       # scene-level, inclusive
    index_frame_original_end:   int       # scene-level, exclusive

    @property
    def length(self) -> int:
        return self.index_frame_original_end - self.index_frame_original_start

    def to_str(self) -> str:
        return '%s/%s/%s pid=%d [%d, %d) (len=%d)' % (
            self.name_dataset, self.name_scene, self.name_seq, self.id_person,
            self.index_frame_original_start, self.index_frame_original_end, self.length,
        )
