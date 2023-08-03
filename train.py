import argparse
import json
import logging
import os
import sqlite3
import sys

from src.training.train import train_loop
from src.utils.db import dict_factory, get_dataset, get_image, get_labels, get_windows

# Configure logger
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s (%(name)s) (%(levelname)s): %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")

    # General
    parser.add_argument(
        "--config_path",
        help="Path to model and training config json.",
        default="./data/cfg/train_sentinel1_detector.json",
    )
    parser.add_argument(
        "--training_data_dir",
        help="Path to training data directory containing preprocess folder.",
        default="./data",
    )
    parser.add_argument(
        "--metadata_path",
        help="Path to sqlite database containing metadata",
        default="./data/metadata.sqlite3",
    )
    parser.add_argument(
        "--save_dir",
        help="Path to train artifact output directory.",
        default="./data/output",
    )

    args = parser.parse_args()
    return args


def main(
    config_path: str, training_data_dir: str, metadata_path: str, save_dir: str
) -> None:
    """Run a training loop for a model.

    Parameters
    ----------
    config_path: str
        Path to configuration json for model one wants to train,
        and data to train on.

    training_data_dir: str
        Path to directory containing training data.

    metadata_path: str
        Path to metadata sqlite file.

    save_dir: str
        Path to directory in which to save train artifacts.
        If nonexistent, will be created.

    Returns
    -------
    : None
    """
    with open(config_path) as f:
        cfg = json.load(f)

    logger.info("Reading training metadata.")

    # Instantiate DB conn
    db_path = os.path.abspath(metadata_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    # Get dataset specified in cfg from db
    dataset_id = cfg["DatasetID"]
    dataset = get_dataset(conn, dataset_id)

    # Get corresponding windows from db
    windows = get_windows(conn, dataset_id)
    windows_with_labels = []

    # Populate labels associated with each window from db
    for window in windows:
        if window["hidden"]:
            continue
        image = get_image(conn, window["image_id"])
        labels = get_labels(conn, window["id"])
        updated_window = window
        updated_window["image"] = image
        updated_window["labels"] = labels
        windows_with_labels.append(updated_window)

    conn.close()

    # Train and validation loop
    train_loop(cfg, dataset, windows_with_labels, save_dir, training_data_dir)
    return None


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
