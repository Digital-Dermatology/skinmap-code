"""Image transformation utilities for SkinMap."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_imagenet_transform():
    """Get standard ImageNet normalization transform for SSL models.

    Returns:
        torchvision.transforms.Compose: Transform pipeline with:
            - Resize to 256x256 with bicubic interpolation
            - Center crop to 224x224
            - Convert to tensor
            - Normalize with ImageNet mean and std
    """
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
