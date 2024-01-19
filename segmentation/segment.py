from collections import Counter
import logging
from typing import (
    Dict,
    List,
)

import cv2
from groundingdino.util.inference import Model
import numpy as np
from segment_anything import (
    SamPredictor,
    sam_model_registry,
)
import supervision as sv
import torch
from tqdm import tqdm

import annotation
from config import CONFIG
import oslib


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        mask, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True,
        )
        index = np.argmax(scores)
        result_masks.append(mask[index])
    return np.array(result_masks)


def save_annotations_to_coco(
    annotations: Dict, images: Dict, annotation_meta: annotation.AnnotationMetadata
):
    """
    Save annotations to a coco json file

    :param annotations: annotations to save, in format:
        {
            "image_name": sv.DetectionAnnotation,
            ...
        }
    :param images: images to save, in format:
        {
            "image_name": np.ndarray,
            ...
        }
    :param annotation_meta: annotation metadata
    """
    if not annotations:
        return

    annotations_path = oslib.gen_unique_filename(
        CONFIG.annotations.coco_annotations_path,
        # uuid="1",
    )
    sv.DetectionDataset(
        classes=CONFIG.ontology.key,
        images=images,
        annotations=annotations,
    ).as_coco(
        images_directory_path=None,  # do not copy images to a new directory
        annotations_path=str(annotations_path),
        min_image_area_percentage=CONFIG.annotations.min_image_area_percentage,
        max_image_area_percentage=CONFIG.annotations.max_image_area_percentage,
        approximation_percentage=CONFIG.annotations.approximation_percentage,
    )

    coco_annotations = annotation.CocoAnnotation(annotations_path)
    # coco_annotations.add_rle_to_annot(annotations)
    coco_annotations.update_image_ids(annotation_meta.last_index)
    coco_annotations.save()
    annotation_meta.save(coco_annotations)


def run() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    annotation_meta = annotation.AnnotationMetadata(
        path=CONFIG.annotations.metadata_path,
    )

    cached_filenames = annotation_meta.get_filenames()

    img_paths = oslib.load_img_paths(
        root_dir=CONFIG.data.image_dir,
        accepted_extensions=CONFIG.data.image_extensions,
        skip_files=cached_filenames,
    )
    img_labels = {
        path.name: annotation.get_labels_dict(path, CONFIG.annotations.labels)
        for path in img_paths
    }
    annotation_meta.add_labels(img_labels)

    if len(img_paths) == 0:
        return 0

    grounding_dino_model = Model(
        model_config_path=str(CONFIG.grounding_dino.config_path),
        model_checkpoint_path=str(CONFIG.grounding_dino.checkpoint_path),
    )

    sam = sam_model_registry[CONFIG.sam.encoder_version](
        checkpoint=CONFIG.sam.checkpoint_path,
    )
    sam_predictor = SamPredictor(sam)

    images = {}
    annotations = {}

    for i, img in enumerate(tqdm(img_paths)):
        image = cv2.imread(str(img))

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CONFIG.ontology.value,
            box_threshold=CONFIG.grounding_dino.box_threshold,
            text_threshold=CONFIG.grounding_dino.text_threshold,
        )

        # detections = detections[detections.class_id != None]
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )
        images[img.name] = image
        annotations[img.name] = detections

        if (i + 1) % CONFIG.checkpoint_step == 0:
            save_annotations_to_coco(
                annotations,
                images,
                annotation_meta,
            )
            images = {}
            annotations = {}

    save_annotations_to_coco(
        annotations,
        images,
        annotation_meta,
    )

    return len(img_paths)
