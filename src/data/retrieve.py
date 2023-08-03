import json
import os
import os.path
import shutil
import typing as t

from src.data.preprocess import preprocess
from src.data.warp import get_image_info, warp
from src.utils.parallel import starmap


class RetrieveImage(object):
    """Container for image metadata."""

    def __init__(self, uuid, name, time, channels, format="geotiff", pixel_size=None):
        self.uuid = uuid
        self.name = name
        self.time = time
        self.channels = channels
        self.format = format
        self.pixel_size = pixel_size


def warp_and_print_one(
    image: RetrieveImage,
    tmp_dir: str,
    channel: dict,
    dst_dir: str = "../data/images/",
    roi: dict = None,
) -> dict:
    """Warp and single RetrieveImage instance.

    Parameters
    ----------
    image: RetrieveImage

    tmp_dir: str
        Directory in which intermediate warp artifacts will get stored.
        Intended use is to use a temporary directory.

    channel: dict

    dst_dir: str
        Path where warped images will get written.

    roi: dict
        Dict representation of ROI object as in sqlite metadata database.

    Returns
    -------
    image_data: dict
        Dictionary describing the warped image metadata.
    """
    dst_path = os.path.join(dst_dir, image.uuid)
    os.makedirs(dst_path, exist_ok=True)

    projection = None
    if roi and roi["Projection"]:
        projection = roi["Projection"]

    src_fname = channel["Path"]

    # Determine where to write layer.
    # Projected images don't need to be stored since we can transform coordinates based on the projection details.
    # We also decide whether to warp the image or just copy it.
    if projection:
        layer_fname = os.path.join(
            tmp_dir, "{}_{}.tif".format(image.uuid, channel["Name"])
        )
        image_info = warp(image, src_fname, layer_fname, projection=projection)
    else:
        layer_fname = os.path.join(dst_path, "{}.tif".format(channel["Name"]))
        if image.format == "geotiff":
            shutil.move(src_fname, layer_fname)
            image_info = get_image_info(layer_fname)
        else:
            image_info = warp(image, src_fname, layer_fname)

    image_data = {
        "UUID": image.uuid,
        "Name": image.name,
        "Format": "geotiff",
        "Channels": image.channels,
        "Width": image_info.width,
        "Height": image_info.height,
        "Bounds": image_info.bounds,
        "Time": image.time.isoformat(),
        "Projection": image_info.projection,
        "Column": image_info.column,
        "Row": image_info.row,
        "Zoom": image_info.zoom,
    }

    preprocess(image_data, layer_fname, channel, roi=roi)

    if projection:
        # We don't need the layer, delete it immediately.
        os.remove(layer_fname)

    return image_data


def warp_and_print(
    images: t.List[RetrieveImage], tmp_dir: str, roi: dict = None, workers: int = 0
) -> None:
    """Warp a list of RetrieveImage instances.


    Given a list of RetrieveImage:
    (1) Copy/warp the image file to data/images/ if needed (i.e., if no projection is set and we need the image for coordinate conversion)
    (2) Pre-process the image into tiles.
    (3) Print out image JSON so caller can add it to database.

    Parameters
    ----------
    images: list[RetrieveImage]
        List of RetrieveImage objects to warp.

    tmp_dir: str
        Directory in which intermediate warp artifacts will get stored.
        Intended use is to use a temporary directory.

    roi: dict
        Dict representation of ROI object as in sqlite metadata database.

    workers: int
        Number of workers to use to perform jobs in parallel.

    Returns
    -------
    : None
    """

    # Get list of jobs. Each job is to warp/preprocess one channel of one image.
    jobs = []
    for image in images:
        for channel in image.channels:
            jobs.append((image, tmp_dir, channel, roi))

    image_datas = starmap(warp_and_print_one, [job for job in jobs], workers=workers)

    # image_datas includes one dict per channel of each image.
    # Here, we print an arbitrary dict per image.
    seen_uuids = set()
    for image_data in image_datas:
        if image_data["UUID"] in seen_uuids:
            continue
        seen_uuids.add(image_data["UUID"])
        print("imagejson:" + json.dumps(image_data), flush=True)
