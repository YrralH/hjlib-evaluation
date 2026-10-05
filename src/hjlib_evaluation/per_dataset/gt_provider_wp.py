'''WorldPose compatibility wrapper around the full-label SMPL GT provider.'''
from hjlib_evaluation.dumped_smpl_gt_provider import Dumped_SMPL_GT_Provider
from hjlib_evaluation.per_dataset.wp_eval_meta import WP_EVAL_META


class WP_GT_Provider(Dumped_SMPL_GT_Provider):
    name_dataset = 'worldpose_smpl'

    def __init__(self, path_root_label: str) -> None:
        super().__init__(self.name_dataset, path_root_label, WP_EVAL_META)
