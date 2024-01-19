import logging
from PIL import Image
import random
from typing import (
    Protocol,
    Tuple,
    Union,
)

import cv2
import numpy as np

from torchvision import transforms


class Transformation(Protocol):
    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class DummyTransformation:
    def __call__(self, image, mask):
        return image, mask


class BlurTransformation:
    def __init__(
        self,
        kernel_size: int = 15,
        inverse=False,
        probability=1.0,
    ):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size
        self.inverse = inverse
        self.probability = probability

    def __call__(self, image, mask):
        random_chance = random.random()

        if random_chance < self.probability:
            if self.inverse:
                mask = ~mask
            blurred_image = cv2.GaussianBlur(
                image,
                (self.kernel_size, self.kernel_size),
                0,
            )

            return np.where(mask[:, :, None], blurred_image, image), mask
        return image, mask


class SolidColorTransformation:
    def __init__(self, color: Tuple[int, int, int] = (0, 0, 0), inverse=False):
        self.color = color
        self.inverse = inverse

    def __call__(self, image, mask):
        if self.inverse:
            mask = ~mask
        return np.where(mask[:, :, None], self.color, image), mask


# class GrayscaleTransformation:
#     def __init__(self, inverse=False):
#         self.inverse = inverse
#         self.transform = transforms.Grayscale(3)

#     def __call__(self, image: Union[np.ndarray, Image.Image], mask):
#         if self.inverse:
#             mask = ~mask

#         if not isinstance(image, Image.Image):
#             image = Image.fromarray(image.astype("uint8"))

#         # Calculate grayscale using luminosity method
#         # grayscale = np.dot(image[..., :3], [0.21, 0.72, 0.07])

#         return (
#             np.where(
#                 mask[:, :, None], np.array(self.transform(image)), np.array(image)
#             ),
#             mask,
#         )


# class ColorJitterTransformation:
#     def __init__(
#         self,
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.1,
#         inverse=False,
#     ):
#         self.inverse = inverse
#         self.transform = transforms.ColorJitter(
#             brightness=brightness,
#             contrast=contrast,
#             saturation=saturation,
#             hue=hue,
#         )

#     def __call__(self, image: Union[np.ndarray, Image.Image], mask):
#         if self.inverse:
#             mask = ~mask

#         if not isinstance(image, Image.Image):
#             image = Image.fromarray(image.astype("uint8"))

#         return (
#             np.where(
#                 mask[:, :, None], np.array(self.transform(image)), np.array(image)
#             ),
#             mask,
#         )


# class RandomTransformation:
#     def __init__(self, inverse=False):
#         self.inverse = inverse
#         self.transform = transforms.RandAugment()

#     def __call__(self, image: Union[np.ndarray, Image.Image], mask):
#         if self.inverse:
#             mask = ~mask

#         if not isinstance(image, Image.Image):
#             image = Image.fromarray(image.astype("uint8"))

#         return (
#             np.where(
#                 mask[:, :, None], np.array(self.transform(image)), np.array(image)
#             ),
#             mask,
#         )


class TorchTransformation:
    def __init__(self, transform, inverse=False, **kwargs):
        self.inverse = inverse
        self.transform = transform(**kwargs)

    def __call__(self, image: Union[np.ndarray, Image.Image], mask):
        if self.inverse:
            mask = ~mask

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype("uint8"))

        return (
            np.where(
                mask[:, :, None], np.array(self.transform(image)), np.array(image)
            ),
            mask,
        )

    @classmethod
    def from_torch_name(cls, name: str, inverse: bool = False, **kwargs):
        if name in ("RandAugment", "rand_augment"):
            return cls(transforms.RandAugment, inverse, **kwargs)
        if name in ("AutoAugment", "auto_augment"):
            return cls(transforms.AutoAugment, inverse, **kwargs)
        if name in ("ColorJitter", "color_jitter"):
            return cls(transforms.ColorJitter, inverse, **kwargs)
        if name in ("Grayscale", "grayscale"):
            return cls(transforms.Grayscale, inverse, **kwargs)

        raise ValueError(f"Unsupported torch transformation name: {name}")


def create_transformation(
    transformation_name: str, inverse: bool, **kwargs
) -> Transformation:
    if transformation_name in ("blur", "BlurTransformation"):
        return BlurTransformation(inverse=inverse, **kwargs)
    if transformation_name in ("solid_color", "SolidColorTransformation"):
        return SolidColorTransformation(inverse=inverse, **kwargs)
    if transformation_name in ("dummy", "DummyTransformation"):
        return DummyTransformation()
    if transformation_name in ("rand_augment", "RandAugmentTransformation"):
        return TorchTransformation.from_torch_name(
            "RandAugment", inverse=inverse, **kwargs
        )
    if transformation_name in ("auto_augment", "AutoAugmentTransformation"):
        return TorchTransformation.from_torch_name(
            "AutoAugment", inverse=inverse, **kwargs
        )
    if transformation_name in ("color_jitter", "ColorJitterTransformation"):
        return TorchTransformation.from_torch_name(
            "ColorJitter", inverse=inverse, **kwargs
        )
    if transformation_name in ("grayscale", "GrayscaleTransformation"):
        return TorchTransformation.from_torch_name(
            "Grayscale", inverse=inverse, **kwargs
        )

    raise ValueError(f"Unsupported transformation name: {transformation_name}")
