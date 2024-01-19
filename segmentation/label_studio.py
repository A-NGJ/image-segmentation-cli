import logging
from typing import (
    List,
    Union,
)

from label_studio_converter.brush import (
    decode_rle,
)
import numpy as np
import requests

from config import CONFIG
import oslib

MASK_SCALING_FACTOR = 255


def rle2mask(rle: Union[str, List[int]], shape: tuple) -> np.ndarray:
    """
    Converts RLE, that uses 4-channel encoding scheme, to a binary mask.

    :param rle: RLE to convert.
    :param shape: Shape of the resulting mask.
    :return: Binary mask.
    """

    mask_1d = decode_rle(rle)
    mask_1d = mask_1d[::4]

    mask = mask_1d.reshape(shape)
    return mask.astype(bool)


def get_all_file_uploads(
    token: str,
    project_id: int,
    page_size: int = 100,
    base_url: str = "http://localhost:8080",
):
    file_uploads = []

    headers = {"Authorization": f"Token {token}"}
    params = {
        "page": 1,
        "page_size": page_size,
    }

    while True:
        resp = requests.get(
            f"{base_url}/api/projects/{project_id}/tasks/",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()

        logging.debug(resp.json())
        file_uploads.extend(item["data"]["image"] for item in resp.json())
        if len(resp.json()) < page_size:
            break

        params["page"] += 1

    oslib.write_json(CONFIG.label_studio.file_uploads_cache_path, file_uploads)
    return file_uploads
