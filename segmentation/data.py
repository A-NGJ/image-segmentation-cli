import logging
import random
from typing import (
    Callable,
    Sequence,
    Optional,
    Tuple,
)
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch

from segmentation.annotation import AnnotationMetadata, LABEL_FALL

from segmentation.transformations import (
    Transformation,
    DummyTransformation,
)


def random_split(
    dataset: "SegmentationDataset", validation_split: float = 0.2, group_size: int = 1
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into a training and validation set.

    :param dataset: The dataset to split.
    :param validation_split: The fraction of the dataset to use for validation.
    :param group_size: The number of images to group together.
    :return: The training and validation datasets.
    """
    dataset_size = len(dataset)
    groupped_indices = [
        list(range(i, min(i + group_size, dataset_size)))
        for i in range(0, dataset_size, group_size)
    ]

    random.shuffle(groupped_indices)

    validation_size = int(validation_split * len(groupped_indices))

    validation_groups = groupped_indices[:validation_size]
    training_groups = groupped_indices[validation_size:]

    validation_indices = [i for group in validation_groups for i in group]
    training_indices = [i for group in training_groups for i in group]

    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata: AnnotationMetadata,
        segmentation_transforms: Sequence[Transformation] = (DummyTransformation(),),
        objects: Sequence[Sequence[str]] = (),
        transforms: Optional[Callable] = None,
        target_label: str = LABEL_FALL,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        if len(segmentation_transforms) != len(objects):
            raise ValueError(
                "The number of segmentation transforms must match the number of object groups"
            )

        self.metadata = metadata
        self.segmentation_transforms = segmentation_transforms
        self.transforms = transforms
        self.objects = objects
        self.target_label = target_label
        self.image_size = image_size

    def __len__(self):
        return len(self.metadata.images)

    def __getitem__(self, idx):
        image_meta = self.metadata.images[idx]
        if image_meta.coco_image.image is None:
            image_meta.coco_image.load_image()

        if self.image_size is not None:
            image_meta.coco_image.resize(self.image_size)

        image_sum = None

        for segmentation_transform, objects in zip(
            self.segmentation_transforms, self.objects
        ):
            try:
                image, mask = image_meta.coco_image.mask_objects(
                    objects, segmentation_transform
                )
                if image_sum is None:
                    image_sum = image
                else:
                    image_sum = np.where(mask[:, :, None], image, image_sum)
            except ValueError as err:
                logging.error(
                    f"Failed to load {image_meta.coco_image.file_name}: {err}"
                )
                return (
                    torch.zeros((3, 256, 256)),
                    torch.zeros((1,)),
                    torch.zeros((3, 256, 256)),
                )
        if image_sum is None:
            raise ValueError(
                "Empty image, provide at least one segmentation transform. Use DummyTransformation if you don't want to transform the image."
            )

        image = Image.fromarray(image_sum.astype("uint8"))

        if self.transforms:
            image = self.transforms(image)

        label = image_meta.labels[self.target_label]

        return (
            image,
            self.metadata.map_label_int(self.target_label)[label],
            image_meta.coco_image.file_name,
        )
