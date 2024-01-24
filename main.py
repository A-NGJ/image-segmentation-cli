from argparse import ArgumentParser
import logging
from pathlib import Path
import sys
import warnings


from collections_operations import check_strings_in_list
from config.config import CONFIG
import oslib
from segmentation import (
    annotation,
    label_studio,
    segment,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logging.basicConfig(level=CONFIG.logging_level)

SEGMENT_MODE = "segment"
PARSE_LABEL_STUDIO_MODE = "parse-label-studio"
EXPORT_LABEL_STUDIO_MODE = "export-to-label-studio"


def main(args):
    if args.mode == SEGMENT_MODE:
        if args.clean:
            clean = oslib.yes_or_no(
                question="Are you sure you want to clean? This process is irreversible.",
                default="no",
            )
            if clean:
                oslib.clean_path(CONFIG.annotations.label_studio_path.parent)
                oslib.clean_path(CONFIG.annotations.coco_annotations_path.parent)
                oslib.clean_path(CONFIG.annotations.metadata_path)
        annotations_path = segment.run(CONFIG)
        if not annotations_path:
            logging.info("No new images to segment")

    if args.mode == EXPORT_LABEL_STUDIO_MODE:
        if CONFIG.label_studio.token == "":
            logging.error(
                "Label studio token is not set. Did you set LABEL_STUDIO_TOKEN environment variable?"
            )
            return 1
        if not args.run_id:
            try:
                args.run_id = oslib.get_run_id(args.coco_annotations_path.name)
            except ValueError as err:
                logging.error(err)
                return 1
        label_studio_collection = (
            annotation.LabelStudioAnnotationCollection.from_coco_json(
                path=args.coco_annotations_path,
            )
        )
        file_uploads = oslib.read_json(
            CONFIG.label_studio.file_uploads_cache_path, empty_ok=True
        )
        if not check_strings_in_list(
            label_studio_collection.get_filenames(), file_uploads
        ):
            file_uploads = label_studio.get_all_file_uploads(
                token=CONFIG.label_studio.token,
                project_id=CONFIG.label_studio.project_id,
                page_size=CONFIG.label_studio.page_size,
                base_url=CONFIG.label_studio.base_url,
            )
        else:
            logging.info("Using cached file uploads")
        label_studio_collection.update_file_names(file_uploads)
        label_studio_collection.to_json(
            oslib.gen_unique_filename(
                CONFIG.annotations.label_studio_path,
                uuid=args.run_id,
            )
        )

    if args.mode == PARSE_LABEL_STUDIO_MODE:
        label_studio_collection = annotation.LabelStudioAnnotationCollection.from_json(
            args.file,
        )
        metadata = annotation.AnnotationMetadata(
            CONFIG.annotations.root_path,
        )
        try:
            label_studio_collection.load_metadata(metadata)
            label_studio_collection.update_annotations_with_metadata(args.run_id)
        except ValueError as err:
            logging.error(err)
            return 1
        label_studio_collection.to_coco_json(
            oslib.gen_unique_filename(
                CONFIG.annotations.export_annotations_path,
                uuid=args.run_id or CONFIG.run_id,
            ),
            coco_annotations=annotation.CocoAnnotation(
                args.coco_annotations_path,
            ),
        )

    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    segment_parser = subparsers.add_parser(SEGMENT_MODE)
    segment_parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Clean old annotations and metadata.",
    )
    parse_label_studio_parser = subparsers.add_parser(PARSE_LABEL_STUDIO_MODE)
    parse_label_studio_parser.add_argument(
        "file",
        type=Path,
        help="Exported Label Studio annotations file",
    )
    parse_label_studio_parser.add_argument(
        "--coco-path",
        dest="coco_annotations_path",
        type=Path,
        help="Path to COCO annotations file. Used to retrieve additioal information about annotations in a pipeline",
    )
    parse_label_studio_parser.add_argument(
        "--run-id",
        type=str,
        help="Run UUID.",
    )
    export_to_label_studio_parser = subparsers.add_parser(EXPORT_LABEL_STUDIO_MODE)
    export_to_label_studio_parser.add_argument(
        "coco_annotations_path",
        type=Path,
        help="Path to COCO annotations file",
    )
    export_to_label_studio_parser.add_argument(
        "--run-id",
        type=str,
        help="Run UUID.",
    )
    args = parser.parse_args()

    sys.exit(main(args))
