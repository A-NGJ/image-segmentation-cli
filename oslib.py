from functools import partial
import json
import logging
from pathlib import Path
import re
import sys
from typing import (
    Collection,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from uuid import uuid4
import yaml
from custom_types import StrPath


def list_directory(path: Path, glob_pattern: str) -> List[Path]:
    return list(path.glob(glob_pattern))


def gen_unique_filename(filename: Path, uuid: str = "") -> Path:
    """Generate a new annotations filename by appending uuid"""

    if not uuid:
        uuid = str(uuid4())[0:8]
    return filename.parent / f"{filename.stem}-{uuid}{filename.suffix}"


def clean_path(path: Path, recursive=False) -> None:
    """
    Remove all files from a path

    :param path: path to clean
    :param recursive: if True, recursively remove all files from subdirectories
    """
    if not path.exists():
        logging.info(f"Path {path} does not exist, cannnot be removed")
        return

    if path.is_file():
        path.unlink()
        return

    if path.is_dir():
        glob_method = path.rglob if recursive else path.glob
        for item in glob_method("*"):
            if item.is_file():
                item.unlink()


def load_img_paths(
    root_dir: Path,
    accepted_extensions: Union[List[str], Set[str]] = {".jpg", ".png", ".jpeg"},
    skip_files: Union[List[str], Set[str]] = set(),
) -> List[Path]:
    img_paths = []
    for path in sorted(root_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix in accepted_extensions
            and path.name not in skip_files
        ):
            img_paths.append(path)

    return img_paths


def yes_or_no(question: str, default="") -> bool:
    """Ask a yes/no question via input() and return their answer.

    :param question: string that is presented to the user.
    :param default: is used if the user hits <Enter>. It must be
                    "yes" (the default), "no" or None (meaning
                    an answer is required of the user).
    :return: True for "yes" or False for "no".

    Source: https://stackoverflow.com/a/3041990/1205815
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if default == "":
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"Invalid default answer: '{default}'")

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()

        if default is not None and choice == "":
            return valid[default]
        if choice in valid:
            return valid[choice]
        sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def _check_for_io_errors(path: Path, suffix: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    if path.suffix != suffix:
        raise ValueError(f"File {path} is not a {suffix} file")


def read_json(path: Union[str, Path], empty_ok: bool = False) -> Union[List, Dict]:
    """Read json file"""
    path = Path(path)

    if empty_ok and not path.exists():
        return {}

    _check_for_io_errors(path, ".json")

    with path.open("r") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError:
            logging.error(f"Failed to load json from {path}")
            raise


def write_json(
    path: Union[str, Path],
    data: Union[List, Dict],
    indent=None,
    create_parents: bool = False,
    exists_ok: bool = False,
    overwrite: bool = False,
) -> None:
    """Write json file"""
    path = Path(path)

    if path.suffix != ".json":
        raise ValueError(f"File {path} is not a .json file")

    if path.exists():
        if not exists_ok:
            raise FileExistsError(f"File {path} already exists")
        if not overwrite:
            path = gen_unique_filename(path)
        else:
            logging.warning(f"File {path} already exists, overwriting")

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=indent)


def read_yaml(path: Union[str, Path], empty_ok: bool = False) -> Union[List, Dict]:
    """Read yaml file"""
    path = Path(path)

    if empty_ok and not path.exists():
        return {}

    _check_for_io_errors(path, ".yaml")

    with path.open("r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            logging.error(f"Failed to load yaml from {path}")
            raise


def get_run_id(s: str) -> str:
    """Get run id from a string"""

    uuid = s.split("-")[-1]
    uuid_match = re.match(r"[a-f0-9]{8}", uuid)
    if not uuid_match:
        raise ValueError(f"Failed to find run id in {s}")
    return uuid_match.group(0)


def list_path(
    path: StrPath,
    accepted_extensions: Collection[str],
    exclude_extensions: Optional[Collection[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a given path. If path is a file, return a list with the file.

    :param path: Path to the file/directory.
    :param accepted_extensions: List of accepted extensions.
    :param exclude_extensions: List of excluded extensions.
    """
    if isinstance(path, str):
        path = Path(path)

    if exclude_extensions is None:
        exclude_extensions = set()

    if path.is_file():
        if path.suffix in accepted_extensions and path.suffix not in exclude_extensions:
            return [path]
        logging.warning(f"File {path} is of unsupported type.")
        return []

    iter_func = partial(path.rglob, "*") if recursive else partial(path.iterdir)

    return [
        p
        for p in iter_func()
        if p.is_file()
        and p.suffix in accepted_extensions
        and p.suffix not in exclude_extensions
    ]
