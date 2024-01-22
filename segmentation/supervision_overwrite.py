"""
This file overrides detections_to_coco_annotations() function from supervision==0.17.1 package.
It adds RLE encoding to the output COCO annotations. Look at line 58 for the change.

For the original supervision package refer to: https://github.com/roboflow/supervision.
"""

from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from label_studio_converter.brush import mask2rle
from supervision import DetectionDataset
from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    save_dataset_images,
)
from supervision.detection.core import Detections
from supervision.dataset.formats.coco import classes_to_coco_categories
from supervision.utils.file import save_json_file

MASK_SCALING_FACTOR = 255


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    coco_annotations = []
    for xyxy, mask, _, class_id, _ in detections:
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        polygon = []
        if mask is not None:
            polygon = list(
                approximate_mask_with_polygons(
                    mask=mask,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )[0].flatten()
            )
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": box_width * box_height,
            "segmentation": [polygon] if polygon else [],
            "rle": mask2rle(mask.astype(np.uint8) * MASK_SCALING_FACTOR)
            if mask is not None
            else [],
            "iscrowd": 0,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def save_coco_annotations(
    annotation_path: str,
    images: Dict[str, np.ndarray],
    annotations: Dict[str, Detections],
    classes: List[str],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
    info = {}
    licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0",
        }
    ]

    coco_annotations = []
    coco_images = []
    coco_categories = classes_to_coco_categories(classes=classes)

    image_id, annotation_id = 1, 1
    for image_path, image in images.items():
        image_height, image_width, _ = image.shape
        image_name = f"{Path(image_path).stem}{Path(image_path).suffix}"
        coco_image = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        coco_images.append(coco_image)
        detections = annotations[image_path]

        coco_annotation, annotation_id = detections_to_coco_annotations(
            detections=detections,
            image_id=image_id,
            annotation_id=annotation_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        coco_annotations.extend(coco_annotation)
        image_id += 1

    annotation_dict = {
        "info": info,
        "licenses": licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    save_json_file(annotation_dict, file_path=annotation_path)


class DetectionDatasetRLE(DetectionDataset):
    def as_coco(
        self,
        images_directory_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to COCO format. This method saves the
        images and their corresponding annotations in COCO format.

        Args:
            images_directory_path (Optional[str]): The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_path (Optional[str]): The path to COCO annotation file.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of polygon points
                to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
        """
        if images_directory_path is not None:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_path is not None:
            save_coco_annotations(
                annotation_path=annotations_path,
                images=self.images,
                annotations=self.annotations,
                classes=self.classes,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
