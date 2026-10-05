# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR photometric augmentation for SGAD region matching.

import albumentations as A


class DarkAug(object):
    """Apply randomized low-light photometric augmentation."""

    def __init__(self) -> None:
        self.augmentor = A.Compose([
            A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.5, 0.0), contrast_limit=(-0.5, 0.3)),
            # A.Blur(p=0.1, blur_limit=(3, 9)),
            # A.MotionBlur(p=0.2, blur_limit=(3, 25)),
            # A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
            A.HueSaturationValue(p=0.5, val_shift_limit=10)
        ], p=0.5)

    def __call__(self, x):
        return self.augmentor(image=x)['image']


def build_augmentor(method=None, **kwargs):
    if method == 'dark':
        return DarkAug()
    elif method is None:
        return None
    else:
        raise ValueError(f'Invalid augmentation method: {method}')
