import datetime
import logging
import os
import os.path
import shutil
import subprocess

from sentinelsat import SentinelAPI

from src.data.preprocess import preprocess
from src.data.warp import get_image_info, warp
from src.utils.parallel import starmap

logger = logging.getLogger("downloader")


def retrieve(images, tmp_dir, download_dir, workers=0):
    """Retrieve images from ESA API."""
    api = SentinelAPI(
        os.environ.get("COPERNICUS_USERNAME"),
        os.environ.get("COPERNICUS_PASSWORD"),
        "https://scihub.copernicus.eu/dhus",
        timeout=60,
    )

    # Filter for those images which still need to be preprocessed.
    preprocess_dir = os.path.join(download_dir, "preprocess")
    filtered_images = [
        image
        for image in images
        if not os.path.isdir(os.path.join(preprocess_dir, image["uuid"]))
    ]
    logger.info(f"{len(images) - len(filtered_images)}/{len(images)} already processed")

    # For unprocessed targets, download compressed images that have yet to be downloaded
    needed_filenames = [
        image["name"]
        for image in filtered_images
        if not os.path.exists(
            os.path.join(
                download_dir, "image_cache", image["name"].replace(".SAFE", ".zip")
            )
        )
    ]
    needed_product_ids = [list(api.query(filename=fn))[0] for fn in needed_filenames]
    api.download_all(
        needed_product_ids,
        directory_path=os.path.join(download_dir, "image_cache"),
        n_concurrent_dl=4,
        max_attempts=4,
    )

    # Pass image metadata to be inflated and preprocessed
    image_paths = {
        image["uuid"]: os.path.join(
            download_dir, "image_cache", image["name"].replace(".SAFE", ".zip")
        )
        for image in filtered_images
    }
    return retrieve_paths(image_paths, tmp_dir, workers=workers)


def unzip(fname, tmp_dir):
    """Unzip an archive to a specified directory."""
    subprocess.call(["unzip", fname, "-d", tmp_dir])


def retrieve_paths(image_paths, tmp_dir, workers=0):
    """Decompresses and collects metadata associated with image paths."""
    # Unzip.
    starmap(
        unzip, [(fname, tmp_dir) for fname in image_paths.values()], workers=workers
    )

    wanted_channels = set(["vh", "vv"])

    # Add images.
    images = []
    for uuid, path in image_paths.items():
        scene_id = os.path.basename(path).replace(".zip", ".SAFE")
        if not scene_id.endswith(".SAFE"):
            continue

        # Identify which channels are available.
        src_path = os.path.join(tmp_dir, scene_id)
        measurement_path = os.path.join(src_path, "measurement")
        if not os.path.exists(measurement_path):
            logger.error("scene {} missing measurement path".format(scene_id))
            continue

        channels = []
        for fname in os.listdir(measurement_path):
            channel_name = fname.split("-")[3]
            if wanted_channels and channel_name not in wanted_channels:
                continue

            channels.append(
                {
                    "Name": channel_name,
                    "Path": os.path.join(measurement_path, fname),
                    "Count": 1,
                }
            )

        # Extract timestamp from scene ID. We use the first time that appears in it.
        # Should appear like "..._20211101T181712_..."
        parts = scene_id.split("_")
        ts = None
        for part in parts:
            if len(part) != 15 or not part[0:8].isdigit():
                continue

            ts = datetime.datetime(
                year=int(part[0:4]),
                month=int(part[4:6]),
                day=int(part[6:8]),
                hour=int(part[9:11]),
                minute=int(part[11:13]),
                second=int(part[13:15]),
                tzinfo=datetime.timezone.utc,
            )
            break

        images.append(
            RetrieveImage(
                uuid=uuid,
                name=scene_id,
                time=ts,
                format="geotiff",
                channels=channels,
                pixel_size=10,
            )
        )

    return images


class RetrieveImage(object):
    """Metadata container for an image retrieved from ESA api."""

    def __init__(self, uuid, name, time, channels, format="geotiff", pixel_size=None):
        self.uuid = uuid
        self.name = name
        self.time = time
        self.channels = channels
        self.format = format
        self.pixel_size = pixel_size


def warp_and_print_one(image, tmp_dir, channel, download_dir, roi=None):
    """ """
    dst_path = os.path.join(download_dir, "images", image.uuid)
    os.makedirs(dst_path, exist_ok=True)

    projection = "epsg:3857"
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

    preprocess(image_data, layer_fname, channel, roi=roi, dest_dir=download_dir)

    if projection:
        # We don't need the layer, delete it immediately.
        os.remove(layer_fname)

    return image_data


def warp_and_print(images, tmp_dir, download_dir, roi=None, workers=0):
    """
    Given a list of RetrieveImage:
    (1) Copy/warp the image file to data/images/ if needed (i.e., if no projection is set and we need the image for coordinate conversion)
    (2) Pre-process the image into tiles.
    (3) Print out image JSON so caller can add it to database.
    """

    # Get list of jobs. Each job is to warp/preprocess one channel of one image.
    jobs = []
    for image in images:
        for channel in image.channels:
            jobs.append((image, tmp_dir, channel, download_dir, roi))

    image_datas = starmap(warp_and_print_one, [job for job in jobs], workers=workers)

    # image_datas includes one dict per channel of each image.
    # Here, we print an arbitrary dict per image.
    seen_uuids = set()
    for image_data in image_datas:
        if image_data["UUID"] in seen_uuids:
            continue
        seen_uuids.add(image_data["UUID"])
        logger.info(f"Finished preprocessing Image with UUID={image_data['UUID']}")
