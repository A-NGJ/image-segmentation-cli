from dataclasses import (
    dataclass,
    field,
    is_dataclass,
)
import os
from pathlib import Path
from typing import (
    Dict,
    List,
)
from uuid import uuid4

import yaml

ROOT = Path(__file__).parent


def nested_dataclass(*args, **kwargs):
    def wrapper(check_class):
        check_class = dataclass(check_class, **kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = check_class.__annotations__.get(name, None)

                if is_dataclass(field_type) and isinstance(value, dict):
                    kwargs[name] = field_type(**value)
                o_init(self, *args, **kwargs)

        check_class.__init__ = __init__

        return check_class

    return wrapper(args[0]) if args else wrapper


class PathPostInitMixin:
    def path_post_init(self) -> None:
        """Convert fields to type hinted type after init."""
        for field_name, field_type in self.__annotations__.items():
            value = getattr(self, field_name)
            if isinstance(value, str) and field_type == Path:
                setattr(self, field_name, Path(value))

    def __post_init__(self):
        self.path_post_init()


@dataclass
class GroundingDinoConfig(PathPostInitMixin):
    config_path: Path
    checkpoint_path: Path
    box_threshold: float
    text_threshold: float


@dataclass
class SamConfig(PathPostInitMixin):
    checkpoint_path: Path
    encoder_version: str


@dataclass
class OntologyConfig:
    key: List[str]
    value: List[str]


@dataclass
class DataConfig(PathPostInitMixin):
    image_dir: Path
    image_extensions: List[str]


@dataclass
class AnnotationsConfig(PathPostInitMixin):
    root_path: Path
    min_image_area_percentage: float
    max_image_area_percentage: float
    approximation_percentage: float
    coco_annotations_path: Path
    export_annotations_path: Path = ""
    metadata_file: str = "metadata.json"
    label_studio_path: Path = ""
    labels: Dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.coco_annotations_path = self.root_path / self.coco_annotations_path
        self.export_annotations_path = self.root_path / self.export_annotations_path
        self.label_studio_path = self.root_path / self.label_studio_path


@dataclass
class LabelStudioConfig(PathPostInitMixin):
    base_url: str = ""
    project_id: int = 0
    page_size: int = 0
    file_uploads_cache_path: Path = ""
    token: str = ""


@nested_dataclass
class AppConfig:
    grounding_dino: GroundingDinoConfig
    sam: SamConfig
    logging_level: str
    ontology: OntologyConfig
    data: DataConfig
    annotations: AnnotationsConfig
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    checkpoint_step: int = 100
    run_id: str = ""

    def __post_init__(self):
        self.run_id = str(uuid4())[:8]

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            return cls(**yaml.safe_load(f))


CONFIG_PATH_ENV = "IMAGE_SEGMENTATION_CONFIG_PATH"
TOKEN_ENV = "LABEL_STUDIO_TOKEN"
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

config_path = Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
token = os.environ.get(TOKEN_ENV, "")
CONFIG = AppConfig.from_yaml(config_path)
CONFIG.label_studio.token = token
