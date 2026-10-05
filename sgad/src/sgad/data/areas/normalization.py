import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet_images(images: torch.Tensor) -> torch.Tensor:
    """Apply the ImageNet normalization expected by the DINO backbones.

    Args:
        images: RGB image tensor in the ``[0, 1]`` range with shape
            ``[C, H, W]`` or ``[B, C, H, W]``.
    """
    shape = [1] * images.ndim
    shape[-3] = 3
    mean = images.new_tensor(IMAGENET_MEAN).view(shape)
    std = images.new_tensor(IMAGENET_STD).view(shape)
    return (images - mean) / std
