import argparse
import logging
import os
import sqlite3
import sys
import tempfile
import typing as t

from src.utils.db import dict_factory, get_image, get_windows
from src.utils.download.sentinel1 import retrieve, warp_and_print

# Configure logger
logger = logging.getLogger("downloader")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s (%(name)s) (%(levelname)s): %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Raw imagery download script.")

    # General
    parser.add_argument(
        "--save_dir",
        help="Path to dir where data will be saved.",
        default="./training_data",
    )
    parser.add_argument(
        "--db_path",
        help="Path to sqlite DB with labels",
        default="./data/metadata.sqlite3",
    )

    args = parser.parse_args()
    return args


def get_corners(im: dict) -> t.Tuple[float, float, float, float]:
    """Get min and max corner coords of image.

    Parameters
    ----------
    im: dict
        Dictionary w/ column, row, width and height keys.

    Returns
    -------
    x_min: float
        Minimal x coordinate.

    y_min: float
        Minimal y coordinate.

    x_max: float
        Maximal x coordinate.

    y_max: float
        Maximal y coordinate.
    """
    x_min, y_min = im["column"], im["row"]
    x_max, y_max = im["column"] + im["width"], im["row"] + im["height"]
    return x_min, y_min, x_max, y_max


def has_intersection(im1: dict, im2: dict) -> bool:
    """Check whether two image records have nonempty intersection.

    Parameters
    ----------
    im1: dict
        Dictionary for one image w/ column, row, width and height keys.

    im2: dict
        Dictionary for second image w/ column, row, width and height keys.

    Returns
    -------
    : bool
        True if the images have nonempty intersection.
    """
    x1_min, y1_min, x1_max, y1_max = get_corners(im1)
    x2_min, y2_min, x2_max, y2_max = get_corners(im2)

    return x2_min < x1_max and x1_min < x2_max and y2_min < y1_min and y1_min < y2_max


def produce_overlaps(
    images: t.List[dict], overlap_options: t.List[dict], save_dir: str
) -> None:
    """Produce overlaps for preprocessed images, if not already populated.

    Parameters
    ----------
    images: list[dict]
        List of image dictionaries, where image dictionaries correspond to image schema in metadata sqlite.

    overlap_options: list[dict]
        List of image dictionaries, where image dictionaries correspond to image schema in metadata sqlite.

    save_dir: str
        Directory in which preprocess data lives.

    Returns
    -------
    : None
    """
    # Produce overlaps for preprocessed images, if not already populated.
    targets = images
    chip_size = 512
    count = 2
    channels = ["vv", "vh"]
    for channel in channels:
        for target in targets:
            # Filter options for those which are not the image
            # in question, but overlap the image in question.
            filtered_overlap_options = []
            for option in overlap_options:
                if option["id"] == target["id"]:
                    continue
                if not has_intersection(target, option):
                    continue
                filtered_overlap_options.append(option)

            if len(filtered_overlap_options) > 0:
                # Account Tiles
                # Produces available_images, a map from (column, row) specifying a
                # tile, to corresponding list of image indices that contain
                # that tile.
                available_images = {}
                for idx, image in enumerate(filtered_overlap_options):
                    hits = {}
                    # get available tiles for image, channels
                    files = os.listdir(
                        os.path.join(save_dir, f"preprocess/{image['uuid']}/{channel}")
                    )
                    for file in files:
                        if ".png" not in file:
                            continue
                        parts = file.split(".png")[0].split("_")
                        column = int(parts[0])
                        row = int(parts[1])
                        hits[(column, row)] = hits.get((column, row), 0) + 1
                    for key, count in hits.items():
                        available_images[key] = available_images.get(key, [])
                        available_images[key].append(idx)
                # Get bounds of current image, in terms of chips
                min_tile = (target["column"] // chip_size, target["row"] // chip_size)
                max_tile = (
                    (target["column"] + target["width"] - 1) // chip_size,
                    (target["row"] + target["height"] - 1) // chip_size,
                )

                # Create overlap channels
                logger.info(
                    f"Creating overlap {channel} channels for image w/ uuid={target['uuid']}."
                )
                for count_idx in range(count):
                    syms = 0
                    channel_name = f"{channel}_overlap{count_idx}"
                    logger.info(f"Creating overlap channel: {channel_name}.")
                    if "overlap" in channel_name and channel in channel_name:
                        if not os.path.isdir(
                            os.path.join(
                                save_dir, f"preprocess/{target['uuid']}/{channel_name}"
                            )
                        ):
                            os.mkdir(
                                os.path.join(
                                    save_dir,
                                    f"preprocess/{target['uuid']}/{channel_name}",
                                )
                            )
                            x = min_tile[0]
                            y = min_tile[1]
                            while x < max_tile[0]:
                                while y < max_tile[1]:
                                    cur_options = available_images.get((x, y), [])
                                    if len(cur_options) < count_idx + 1:
                                        y += 1
                                        continue
                                    option = filtered_overlap_options[
                                        cur_options[count_idx]
                                    ]
                                    src = os.path.join(
                                        save_dir,
                                        "preprocess",
                                        option["uuid"],
                                        channel,
                                        f"{x}_{y}.png",
                                    )
                                    dst = os.path.join(
                                        save_dir,
                                        "preprocess",
                                        target["uuid"],
                                        channel_name,
                                        f"{x}_{y}.png",
                                    )
                                    os.symlink(src, dst)
                                    syms += 1
                                    y += 1
                                x += 1
                            logger.info(f"Created {syms} symlinks.")
                        else:
                            logger.info(
                                f"Overlap channel {channel_name} already exists."
                            )
                    count_idx += 1

    return None


def main(save_dir: str, db_path: str) -> None:
    """Run imagery download script for training data.

    Parameters
    ----------
    save_dir: str
        Path to directory in which downloaded data will be stored.

    db_path: str
        Path to metadata sqlite file defining images to be downloaded.

    Returns
    -------
    : None
    """

    # Create training data dir
    if not os.path.exists(save_dir):
        # logger.debug(f"Creating data dir at {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    # Instantiate DB conn
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    # Get fixed list of windows corresponding to splits/datasets we care about
    dataset_id = 1
    splits = [
        "jan-march-may-2022-point-train",
        "jan-march-may-2022-point-val",
        "jun-july-aug-2022-point-train",
        "jun-july-aug-2022-point-val",
        "apr-2022-point-train",
        "apr-2022-point-val",
        "nov-2021-point-train",  # In long term storage on Copernicus Hub at time of writing
        "nov-2021-point-val",  # In long term storage on Copernicus Hub at time of writing
        "jun-2020-point-train",  # In long term storage on Copernicus Hub at time of writing
        "jun-2020-point-val",  # In long term storage on Copernicus Hub at time of writing
    ]
    windows = []
    for split in splits:
        windows.extend(get_windows(conn, dataset_id, split=split))
    logger.info(f"Collecting images corresponding to {len(windows)} windows.")

    # Get fixed list of corresponding images, and their uuids
    image_ids = set([win["image_id"] for win in windows])
    images = [get_image(conn, id) for id in image_ids]
    logger.info(f"Found {len(images)} such images to synchronize to local.\n")

    # Loop over targets in batches of fixed size, and download and preprocess
    def batcher(lst, batch_size):
        return (lst[pos: pos + batch_size] for pos in range(0, len(lst), batch_size))

    batch_size = 8
    workers = 4
    synced = 0
    for tgt_batch in batcher(images, batch_size=batch_size):
        logger.info(f"Currently syncing {len(tgt_batch)} images...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            retrieved_images = retrieve(tgt_batch, tmp_dir, save_dir, workers=workers)
            warp_and_print(
                retrieved_images, tmp_dir, save_dir, roi=None, workers=workers
            )
        logger.info("...done\n")
        synced += len(tgt_batch)
        logger.info(f"{synced}/{len(images)} total images synced.")

    produce_overlaps(images, images, save_dir)

    return None


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
