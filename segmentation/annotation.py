from collections import (
    defaultdict,
    namedtuple,
)
from copy import deepcopy
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
import logging
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import cv2
from matplotlib.cm import get_cmap
import numpy as np

from custom_types import StrPath
from segmentation.transformations import (
    Transformation,
    SolidColorTransformation,
)
from segmentation.label_studio import rle2mask
import oslib

LABEL_FALL = "fall"


class Keys(str, Enum):
    CATEGORIES = "categories"
    ANNOTATIONS = "annotations"
    ANNOTATION_FILE = "annotation_file"
    LABELS = "labels"
    FILE_NAME = "file_name"
    ID = "id"
    IMAGES = "images"
    IMAGE = "image"
    NAME = "name"
    CATEGORY_ID = "category_id"
    IMAGE_ID = "image_id"
    LAST_INDEX = "last_index"
    RLE = "rle"
    WIDTH = "width"
    HEIGHT = "height"
    BBOX = "bbox"
    FILE_UPLOAD = "file_upload"
    LICENSE = "license"
    LICENSES = "licenses"
    DATE_CAPTURED = "date_captured"


def get_labels_list(path: Path) -> List[str]:
    return [p.name for p in path.parents if p != Path(".")]


def get_labels_dict(path, label_map: Dict[str, List[str]]) -> Dict:
    labels_dict = {}
    for label in get_labels_list(path):
        for key, values in label_map.items():
            if label in values:
                labels_dict[key] = label
                break
    return labels_dict


@dataclass
class AnnotationMetaImage:
    annotation_file: str
    coco_image: "CocoImage"
    labels: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            Keys.ANNOTATION_FILE.value: self.annotation_file,
            Keys.LABELS.value: self.labels,
            **self.coco_image.as_dict(),
        }

    def get_attr(self, attr: str) -> Any:
        val = self
        for a in attr.split("."):
            if isinstance(val, dict):
                val = val.get(a)
            else:
                val = getattr(val, a)
        return val

    def filter_attr(self, attr: str, filter: Callable) -> bool:
        return filter(self.get_attr(attr))


class AnnotationMetadata:
    def __init__(
        self,
        root_path: StrPath,
        metadata_filename: StrPath = "metadata.json",
        annotations_dir: StrPath = "annotations/",
        data_dir: StrPath = "data/",
        write_new: bool = False,
        empty: bool = False,
    ):
        self.root_path = Path(root_path)
        if not (data_dir.startswith("/") or data_dir.startswith(".")):
            self.data_dir = self.root_path / data_dir
        else:
            self.data_dir = data_dir
        self.path = self.root_path / metadata_filename

        self._data: Dict = self._load(write_new) if not empty else {}
        self.last_index = self._data.get(Keys.LAST_INDEX, 0)
        self.labels = {}
        self.image_labels = defaultdict(list)
        self.categories = self._data.get(Keys.CATEGORIES, {})
        self.annotations = (
            self._load_coco_annotations(
                self.root_path / annotations_dir,
                self._data[Keys.ANNOTATIONS],
            )
            if not empty
            else []
        )
        self.images = (
            [
                AnnotationMetaImage(
                    coco_image=CocoImage(
                        annotations=self.get_coco_annotation(
                            img[Keys.ANNOTATION_FILE]
                        ).get_annotations(img[Keys.ID]),
                        id=img[Keys.ID],
                        file_name=img[Keys.FILE_NAME],
                        data_dir=self.data_dir,
                        width=img[Keys.WIDTH],
                        height=img[Keys.HEIGHT],
                        license=img.get(Keys.LICENSE, 1),
                        date_captured=img.get(Keys.DATE_CAPTURED, ""),
                    ),
                    annotation_file=img[Keys.ANNOTATION_FILE],
                    labels=img[Keys.LABELS],
                )
                for img in self._data[Keys.IMAGES]
            ]
            if not empty
            else []
        )

        for img in self.images:
            for key, label in img.labels.items():
                if label not in self.image_labels[key]:
                    self.image_labels[key].append(label)

    def _load(self, write_new: bool = False) -> Dict:
        try:
            return dict(oslib.read_json(self.path))
        except FileNotFoundError:
            if write_new:
                logging.warning(f"File {self.path} does not exist, creating new file")
                return {
                    Keys.LAST_INDEX.value: 0,
                    Keys.IMAGES.value: [],
                    Keys.ANNOTATIONS.value: [],
                    Keys.CATEGORIES.value: {},
                }
            raise

    def as_dict(self) -> Dict:
        return {
            Keys.LAST_INDEX.value: self.last_index,
            Keys.IMAGES.value: [img.as_dict() for img in self.images],
            Keys.ANNOTATIONS.value: self.annotations,  # [a.name for a in self.annotations],
            Keys.CATEGORIES.value: self.categories,
        }

    def _load_coco_annotations(self, root, annotation_paths: List[Path]):
        annotations = []
        for path in annotation_paths:
            coco = CocoAnnotation(root / path, self.data_dir)
            annotations.append(coco)
        return annotations

    def map_label_int(self, key: str) -> Dict[str, int]:
        label_map = {}
        for i, val in enumerate(sorted(self.image_labels[key])):
            label_map[val] = i
        return label_map

    def get_coco_annotation(self, name: str) -> "CocoAnnotation":
        for annotation in self.annotations:
            if annotation.name == name:
                return annotation
        raise ValueError(f"Annotation {name} not found")

    def add_labels(self, labels: Dict[str, Dict]) -> None:
        self.labels.update(labels)

    def update(self, coco_annotation: "CocoAnnotation") -> None:
        if coco_annotation.name not in self.annotations:
            self.annotations.append(coco_annotation.name)

        # add reference to annotation file to each image
        for img in coco_annotation.images:
            annotation_meta_image = AnnotationMetaImage(
                coco_image=img,
                annotation_file=coco_annotation.name,
            )
            self.images.append(annotation_meta_image)

        # add labels to images
        for img in self.images:
            if (
                img.coco_image.file_name in self.labels
            ):  # otherwise skip, image labels are already in metadata
                img.labels = self.labels[img.coco_image.file_name]

        self.last_index = max(img.coco_image.id for img in self.images)
        # self.last_index = self._data[Keys.LAST_INDEX]

        self.categories.update(
            {
                category_name: int(category_id)
                for category_name, category_id in coco_annotation.get_class_index().items()
            }
        )

    def to_json(self, path: StrPath) -> None:
        oslib.write_json(path, self.as_dict(), indent=4)

    def filter_images(
        self, filters: Dict[str, Callable], return_copies=False
    ) -> List[AnnotationMetaImage]:
        filtered_images = []
        for img in self.images:
            for key, img_filter in filters.items():
                if not img.filter_attr(key, img_filter):
                    break
            else:
                if return_copies:
                    img = deepcopy(img)
                filtered_images.append(img)

        return filtered_images

    def remove_images(self, filters: Dict[str, Callable]) -> List[AnnotationMetaImage]:
        """
        Remove images that match filters.

        :returns: list of removed images
        """

        filtered_images = self.filter_images(filters)
        self.images = [img for img in self.images if img not in filtered_images]

        return filtered_images

    def merge(
        self, metadata: "AnnotationMetadata", filters: Optional[Dict] = None
    ) -> None:
        """
        Merge two metadata files.
        """
        if filters is not None:
            for key, f in filters.items():
                if f is None:
                    continue
                if not isinstance(f, Callable):
                    logging.warning(
                        f"Filter {key} is not callable, got {type(f)} instead. Skipping the filter"
                    )
                    continue
                self.images = [img for img in self.images if f(img)]

        self.images.extend(metadata.images)
        self.annotations.extend(metadata.annotations)
        self.categories.update(metadata.categories)
        self.last_index = max(self.last_index, metadata.last_index)

    def get_filenames(self) -> List[str]:
        """
        Get saved filenames of segmented images from a metadata file.
        """

        return [img.coco_image.file_name for img in self.images]

    def get_img_index(
        self,
        attr: str = "coco_image.id",
        transform: Callable = lambda x: x,
    ) -> Dict[str, Any]:
        """
        Get indices of filenames of segmented images from a metadata file.
        """
        return {
            img.coco_image.file_name: transform(img.get_attr(attr))
            for img in self.images
        }

    def get_labels(self) -> Dict[str, Dict]:
        """
        :return: dictionary of labels for each image, where key is image filename
        """
        return {img.coco_image.file_name: img.labels for img in self.images}

    def colormap(self, cmap: str = "tab10") -> Dict[str, List[int]]:
        """
        :return: dictionary of colors for each label, where key is label name
        """
        colormap = get_cmap(cmap, 10)
        return {
            category_name: [int(c * 255) for c in colormap(category_id)[:3]]
            for category_name, category_id in self.categories.items()
        }


@dataclass
class CocoLicense:
    id: int
    name: str
    url: str


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str = "common-objects"


@dataclass
class SingleCocoAnnotation:
    id: int
    image_id: int
    category_id: int
    category: CocoCategory
    rle: List[int]
    bbox: List[int]
    rle_width: int = 0
    rle_height: int = 0
    mask: Optional[np.ndarray] = None

    def init_mask(self) -> np.ndarray:
        self.mask = rle2mask(self.rle, (self.rle_height, self.rle_width))

        return self.mask

    def as_dict(self) -> Dict:
        return {
            Keys.ID.value: self.id,
            Keys.IMAGE_ID.value: self.image_id,
            Keys.CATEGORY_ID.value: self.category_id,
            Keys.RLE.value: self.rle,
            Keys.BBOX.value: self.bbox,
        }


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    data_dir: StrPath
    license: int = 1
    date_captured: str = ""
    image: Optional[np.ndarray] = None
    annotations: List[SingleCocoAnnotation] = field(default_factory=list)

    def __post_init__(self):
        for annot in self.annotations:
            annot.rle_width = self.width
            annot.rle_height = self.height

    def as_dict(self) -> dict:
        return {
            Keys.ID.value: self.id,
            Keys.LICENSE.value: self.license,
            Keys.FILE_NAME.value: self.file_name,
            Keys.WIDTH.value: self.width,
            Keys.HEIGHT.value: self.height,
            Keys.DATE_CAPTURED.value: self.date_captured,
        }

    def load_image(self) -> np.ndarray:
        data_path = Path(self.data_dir)
        img_path = data_path / self.file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_path} does not exist")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image.copy()
        return image

    def draw_masks(self, cmap: str = "tab10") -> np.ndarray:
        colormap = get_cmap(cmap, 10)
        image = self.image
        if self.image is None:
            image = self.load_image()
        for annot in self.annotations:
            try:
                mask = rle2mask(annot.rle, (self.height, self.width))
                image[mask] = [c * 255 for c in colormap(annot.category.id)[:3]]
            except ValueError:
                logging.warning(
                    f"Failed to draw mask for annotation {annot.category.name}"
                )
        return image

    def mask_objects(
        self,
        objects: List[str],
        mask_transformation: Transformation = SolidColorTransformation(),
        raise_if_empty: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = self.image
        if image is None:
            image = self.load_image()

        mask_combined = np.zeros((self.height, self.width), dtype=bool)

        if not objects:
            return image, mask_combined

        for annot in self.annotations:
            if annot.category.name in objects:
                mask = annot.init_mask()
                mask = cv2.resize(
                    (mask * 255).astype(np.uint8), (self.width, self.height)
                )
                mask_combined |= mask.astype(bool)

        if not mask_combined.any():
            warning_message = (
                f"Mask for objects {objects} is empty for image {self.file_name}"
            )
            if raise_if_empty:
                raise ValueError(
                    f"Mask for objects {objects} is empty for image {self.file_name}"
                )
            logging.debug(warning_message)

        return mask_transformation(
            cv2.resize(image, (self.width, self.height)),
            mask_combined,
        )

    def get_bbox(self, annot_id: int) -> list:
        """
        Get bbox coordinates for a given image and category id.

        :param image_id: image id
        :param category_id: category id
        :return: bbox coordinates in format [x, y, width, height]
        """

        for annot in self.annotations:
            if annot.id == annot_id:
                return annot.bbox
        return []

    def resize(self, size: Tuple[int, int]) -> np.ndarray:
        self.width, self.height = size

        return cv2.resize(self.image, size)


class CocoAnnotation:
    MASK_SCALING_FACTOR = 255

    def __init__(self, path: StrPath, data_dir: StrPath = "data/"):
        self.path = Path(path)
        self.data_dir = Path(data_dir)
        self._data: Dict = self._load()
        self.name = self.path.name
        self.licenses: List[CocoLicense] = [
            CocoLicense(**l) for l in self._data.get(Keys.LICENSE, [])
        ]
        self.categories: List[CocoCategory] = [
            CocoCategory(**c) for c in self._data.get(Keys.CATEGORIES, [])
        ]
        self.annotations: List[SingleCocoAnnotation] = [
            SingleCocoAnnotation(
                category=self.get_category(a[Keys.CATEGORY_ID]),
                category_id=a[Keys.CATEGORY_ID],
                id=a[Keys.ID],
                image_id=a[Keys.IMAGE_ID],
                rle=a[Keys.RLE],
                bbox=a[Keys.BBOX],
            )
            for a in self._data.get(Keys.ANNOTATIONS, [])
        ]
        self.images: List[CocoImage] = [
            CocoImage(
                annotations=self.get_annotations(d[Keys.ID]),
                data_dir=self.data_dir,
                **d,
            )
            for d in self._data.get(Keys.IMAGES, [])
        ]

    def _load(self) -> Dict:
        try:
            return dict(oslib.read_json(self.path))
        except FileNotFoundError:
            logging.warning(f"File {self.path} does not exist.")
            return {}

    def as_dict(self) -> Dict:
        return {
            Keys.LICENSES.value: [l.__dict__ for l in self.licenses],
            Keys.CATEGORIES.value: [c.__dict__ for c in self.categories],
            Keys.ANNOTATIONS.value: [a.as_dict() for a in self.annotations],
            Keys.IMAGES.value: [i.as_dict() for i in self.images],
        }

    def save(self, replace=False):
        if replace and self.path.exists():
            self.path.unlink()
        oslib.write_json(self.path, self.as_dict())

    def update_image_ids(self, start_id: int) -> int:
        id_mapping = {}  # old_id -> new_id
        for i, img in enumerate(self.images, start=1):
            id_mapping[img.id] = start_id + i
            img.id = start_id + i
        for i, annot in enumerate(self.annotations, start=1):
            annot.image_id = id_mapping[annot.image_id]

        return start_id + len(self.images)

    def get_img(self, id: int) -> CocoImage:
        for img in self.images:
            if img.id == id:
                return img
        raise ValueError(f"Image with id {id} not found")

    def get_category(self, id: int) -> CocoCategory:
        for category in self.categories:
            if category.id == id:
                return category
        return CocoCategory(id=0, name="", supercategory="")

    def get_annotations(self, img_id: int) -> List[SingleCocoAnnotation]:
        return [a for a in self.annotations if a.image_id == img_id]

    def get_img_index(self) -> Dict[str, int]:
        """
        Get indices of filenames of segmented images from a coco annotations file.
        """
        return {img.file_name: int(img.id) for img in self.images}

    def get_annotations_index(self, key: Keys) -> Dict[str, np.ndarray]:
        """
        Get index of chosen key from annotations in a coco annotations file.
        E.g. get index of bbox coordinates for each image or rle segmentation masks for each image.
        """

        return {
            img.file_name: np.array(
                [getattr(annot, key.value) for annot in img.annotations]
            )
            for img in self.images
        }

    def get_reverse_img_index(self) -> Dict[int, str]:
        """
        Get indices of filenames of segmented images from a coco annotations file.
        """
        return {int(img.id): img.file_name for img in self.images}

    def get_class_index(self) -> Dict[str, int]:
        """
        Get indices of class names from a metadata file.
        """
        return {category.name: category.id for category in self.categories}

    def get_annotations_categories(self) -> List[Dict[str, int]]:
        """
        Get categories of annotations in a coco annotations file.

        :return: dictionary of image ids and their corresponding category ids for each annotation.
        """
        return [
            {
                Keys.IMAGE_ID: annot.image_id,
                Keys.CATEGORY_ID: annot.category.id,
            }
            for annot in self.annotations
        ]


class LabelStudioAnnotation:
    DEFAULT_FROM_NAME = "brush"
    DEFAULT_TO_NAME = "image"
    DEFAULT_PREDICTION_TYPE = "brushlabels"
    DEFAULT_ORIGIN = "manual"

    def __init__(self, **kwargs):
        required_keys = [
            Keys.ID.value,
            Keys.IMAGE.value,
            Keys.WIDTH.value,
            Keys.HEIGHT.value,
        ]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key: {key}")

        self.id = kwargs[Keys.ID]
        self.image = kwargs[Keys.IMAGE]
        self.from_name = self.DEFAULT_FROM_NAME
        self.to_name = self.DEFAULT_TO_NAME
        self.file_upload = kwargs.get(Keys.FILE_UPLOAD, "")
        self.width = kwargs[Keys.WIDTH]
        self.height = kwargs[Keys.HEIGHT]
        self.predictions = kwargs.get("annotations") or [{"result": []}]

    @classmethod
    def from_json_dict(cls, json: Dict) -> "LabelStudioAnnotation":
        return cls(
            id=json[Keys.ID],
            image=json["data"][Keys.IMAGE],
            height=json[Keys.ANNOTATIONS][0]["result"][0]["original_height"],
            width=json[Keys.ANNOTATIONS][0]["result"][0]["original_width"],
            annotations=json[Keys.ANNOTATIONS],
            file_upload=json[Keys.FILE_UPLOAD],
        )

    @classmethod
    def from_coco(cls, coco: Dict, ind: int) -> "LabelStudioAnnotation":
        instance = cls(
            id=coco[Keys.IMAGES][ind][Keys.ID],
            image=coco[Keys.IMAGES][ind][Keys.FILE_NAME],
            height=coco[Keys.IMAGES][ind][Keys.HEIGHT],
            width=coco[Keys.IMAGES][ind][Keys.WIDTH],
        )
        instance._add_predictions_from_coco(coco)
        return instance

    def _add_predictions_from_coco(self, coco: Dict) -> None:
        for annotation in coco.get("annotations", []):
            if annotation["image_id"] == self.id:
                category = coco["categories"][annotation["category_id"]]
                self.predictions[0]["result"].append(
                    {
                        Keys.ID: annotation["id"],
                        "type": self.DEFAULT_PREDICTION_TYPE,
                        "value": {
                            "rle": annotation["rle"],
                            "format": "rle",
                            "brushlabels": [category["name"]],
                        },
                        "origin": self.DEFAULT_ORIGIN,
                        "to_name": self.to_name,
                        "from_name": self.from_name,
                        "image_rotation": 0,
                        "original_width": self.width,
                        "original_height": self.height,
                    }
                )

    def as_dict(self):
        return {
            "data": {Keys.IMAGE: self.image},
            "annotations": self.predictions,
        }

    def to_json(self, path: Path = Path("label_studio_annot.json")):
        with open(path, "w") as f:
            json.dump([self.as_dict()], f, indent=4)


AnnotationIdTuple = namedtuple("AnnotationId", ["id", "has_bbox"])


class LabelStudioAnnotationCollection:
    def __init__(
        self,
        annotations: Optional[List[LabelStudioAnnotation]] = None,
    ):
        self.annotations = annotations or []
        self.metadata = None

    def to_dict(self):
        return [annotation.as_dict() for annotation in self.annotations]

    def to_json(self, path: Path = Path("label_studio_annot.json")):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f)
        except EnvironmentError as err:
            logging.error(f"Failed to save annotations to {path}: {err}")

    @classmethod
    def from_coco_dict(cls, coco: Dict):
        return cls(
            annotations=[
                LabelStudioAnnotation.from_coco(coco, ind)
                for ind, _ in enumerate(coco["images"])
            ],
        )

    def to_coco_dict(
        self,
        class_index: Dict[str, int],
        coco_annotations: Optional[CocoAnnotation] = None,
    ) -> Dict:
        coco = {
            "info": {},
            "licenses": [
                {
                    "id": 1,
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "name": "CC BY 4.0",
                }
            ],
            Keys.IMAGES.value: [],
            Keys.ANNOTATIONS.value: [],
            Keys.CATEGORIES.value: [
                {
                    Keys.NAME.value: k,
                    Keys.ID.value: v,
                }
                for k, v in class_index.items()
            ],
        }
        for annotation in self.annotations:
            coco[Keys.IMAGES].append(
                {
                    Keys.ID.value: annotation.id,
                    Keys.FILE_NAME.value: annotation.image,
                    Keys.HEIGHT.value: annotation.height,
                    Keys.WIDTH.value: annotation.width,
                }
            )
            for prediction in annotation.predictions:
                prediction_ids = LabelStudioAnnotationCollection._generate_int_ids(
                    [result[Keys.ID] for result in prediction["result"]]
                )
                for result, prediction_id in zip(prediction["result"], prediction_ids):
                    if result["type"] != "brushlabels":
                        continue
                    category_id = class_index[result["value"]["brushlabels"][0]]

                    coco[Keys.ANNOTATIONS].append(
                        {
                            Keys.ID.value: prediction_id.id,
                            Keys.IMAGE_ID.value: annotation.id,
                            Keys.CATEGORY_ID.value: category_id,
                            Keys.RLE.value: result["value"][Keys.RLE],
                            Keys.BBOX.value: coco_annotations.get_img(
                                annotation.id
                            ).get_bbox(prediction_id.id)
                            if coco_annotations and prediction_id.has_bbox
                            else [],
                        }
                    )
        return coco

    @staticmethod
    def _generate_int_ids(ids: List) -> List[AnnotationIdTuple]:
        int_ids = []
        non_int_ids = []
        for id in ids:
            try:
                id = int(id)
            except ValueError:
                non_int_ids.append(id)
            else:
                int_ids.append(AnnotationIdTuple(id, True))
        for _ in non_int_ids:
            int_ids.append(
                AnnotationIdTuple((1 + max((i.id for i in int_ids), default=0)), False)
            )

        return int_ids

    @classmethod
    def from_coco_json(cls, path: Path):
        with open(path, "r") as f:
            coco = json.load(f)
        return cls.from_coco_dict(coco)

    def load_metadata(self, metadata: AnnotationMetadata):
        if not isinstance(metadata, AnnotationMetadata):
            raise ValueError(
                f"Expected AnnotationMetadata, got {type(metadata)} instead"
            )
        self.metadata = metadata

    def update_annotations_with_metadata(self, run_id: str):
        """
        Update image ids and image names in annotations
        to match the ones in metadata.
        Remove annotations for images that are not in metadata.

        :raises ValueError: if metadata is not set
        """

        if self.metadata is None:
            raise ValueError("Metadata is not set")

        image_index = self.metadata.get_img_index(transform=int)
        annotation_file_index = self.metadata.get_img_index(attr=Keys.ANNOTATION_FILE)
        updated_annotations = []
        for annotation in self.annotations:
            for img_name, img_id in image_index.items():
                annotation_file_run_id = oslib.get_run_id(
                    annotation_file_index[img_name]
                )
                if (
                    img_name in annotation.image
                    and annotation_file_run_id
                    == oslib.get_run_id(annotation.file_upload)
                    and run_id == annotation_file_run_id
                ):
                    annotation.image = img_name
                    annotation.id = img_id
                    updated_annotations.append(annotation)
                    break
        self.annotations = updated_annotations

    def to_coco_json(
        self, path: Path, coco_annotations: Optional[CocoAnnotation] = None
    ):
        """
        Export annotations to a coco json file.

        :param path: path to a coco json file
        :param coco_annotations: In case of exporting annotations from Label Studio, use information
                                 from the original coco annotatins file, e.g. bounding boxes.
        """
        if self.metadata is None:
            raise ValueError("Metadata is not set")

        coco_dict = self.to_coco_dict(self.metadata.categories, coco_annotations)
        oslib.write_json(path, coco_dict, create_parents=True)

    @classmethod
    def from_json(cls, path: Path):
        with path.open("r") as f:
            annotations = json.load(f)
        return cls(
            annotations=[
                LabelStudioAnnotation.from_json_dict(json=a) for a in annotations
            ],
        )

    def update_file_names(self, filenames: List[str]):
        for i, annotation in enumerate(self.annotations):
            for fname in filenames:
                if annotation.image in fname:
                    self.annotations[i].image = fname
                    break

    def get_filenames(self) -> List[str]:
        return [annotation.image for annotation in self.annotations]
