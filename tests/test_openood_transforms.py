import torch
from PIL import Image
from torchvision import transforms

from oge.data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    ConvertRGB,
    make_cifar10_eval_transform,
    make_cifar10_train_transform,
)


def test_eval_transform_is_deterministic_and_uses_cifar10_normalization():
    image = Image.new("L", (41, 35), color=128)
    transform = make_cifar10_eval_transform()

    first = transform(image)
    second = transform(image)

    torch.testing.assert_close(first, second)
    assert first.shape == (3, 32, 32)
    expected = torch.tensor(
        [(128 / 255 - mean) / std for mean, std in zip(CIFAR10_MEAN, CIFAR10_STD)]
    )
    torch.testing.assert_close(first[:, 0, 0], expected)


def test_train_and_eval_transform_component_order_matches_contract():
    train_types = [type(item) for item in make_cifar10_train_transform().transforms]
    eval_types = [type(item) for item in make_cifar10_eval_transform().transforms]

    assert train_types == [
        ConvertRGB,
        transforms.Resize,
        transforms.CenterCrop,
        transforms.RandomHorizontalFlip,
        transforms.RandomCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert eval_types == [
        ConvertRGB,
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
