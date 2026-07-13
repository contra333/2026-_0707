"""OpenOOD v1.5-aligned CIFAR-10 transforms."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ConvertRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def make_cifar10_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            ConvertRGB(),
            transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def make_cifar10_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            ConvertRGB(),
            transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
