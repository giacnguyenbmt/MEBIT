import numpy as np

from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric import functional as F

class Rotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """
    def __init__(
        self,
        k=1,
        always_apply=False,
        p=0.5
    ):
        super(Rotate90, self).__init__(always_apply, p)
        self.k = k

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {"factor": self.k}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self):
        return ()