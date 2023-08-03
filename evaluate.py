import argparse
import json
import logging
import os
import sqlite3
import sys

from src.data.dataset import Dataset
from src.data.image import Channels
from src.data.transforms import get_transform
from src.inference.pipeline import load_model
from src.training.evaluate import evaluate, get_evaluator
from src.training.utils import collate_fn
from src.utils.db import dict_factory, get_dataset, get_image, get_labels, get_windows

import torch

num_loader_workers = 4

# Configure logger
logger = logging.getLogger("evaluate")
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
        "--model_dir",
        help="Path to model and validation config json.",
        default="./data/inference/frcnn_cmp2/3dff445",
    )
    parser.add_argument(
        "--validation_data_dir",
        help="Path to validation data directory containing preprocess folder.",
        default="./data",
    )
    parser.add_argument(
        "--metadata_path",
        help="Path to sqlite database containing metadata",
        default="./data/metadata.sqlite3",
    )

    args = parser.parse_args()
    return args


def main(
    model_dir: str, validation_data_dir: str, metadata_path: str
) -> None:
    """Run a validation pass on specified model.

    Parameters
    ----------
    model_dir: str
        Path to dir containing model weights and configuration json.

    validation_data_dir: str
        Path to directory containing validation data.

    metadata_path: str
        Path to metadata sqlite file.

    Returns
    -------
    : None
    """
    with open(os.path.join(model_dir, "cfg.json")) as f:
        cfg = json.load(f)

    logger.info("Reading validation metadata.")

    # Instantiate DB conn
    db_path = os.path.abspath(metadata_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    # Get dataset specified in cfg from db
    dataset_id = cfg["DatasetID"]
    dataset = get_dataset(conn, dataset_id)

    # Read test splits specified in config
    val_splits = cfg["Options"]["TestSplits"]

    # Get windows associated with val splits
    windows = []
    for split in val_splits:
        windows.extend(get_windows(conn, dataset_id, split=split))
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

    model_cfg = cfg
    options = model_cfg["Options"]
    channels = Channels(model_cfg["Channels"])
    task = dataset["task"]
    model_cfg["Data"] = {}
    if dataset.get("task"):
        model_cfg["Data"]["task"] = dataset["task"]
    if dataset.get("categories"):
        model_cfg["Data"]["categories"] = dataset["categories"]
    batch_size = options.get("BatchSize", 4)
    chip_size = options.get("ChipSize", 0)
    image_size = options.get("ImageSize", 0)
    half_enabled = options.get("Half", True)
    val_transforms = get_transform(cfg, options, options.get("ValTransforms", []))

    val_data = Dataset(
        dataset=dataset,
        windows=windows,
        channels=channels,
        splits=val_splits,
        transforms=val_transforms,
        image_size=image_size,
        chip_size=chip_size,
        valid=True,
        preprocess_dir=os.path.join(validation_data_dir, "preprocess"),
    )

    device = torch.device("cuda")

    val_sampler = torch.utils.data.SequentialSampler(val_data)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_loader_workers,
        collate_fn=collate_fn,
    )

    # instantiate model from weights and config
    model = load_model(
        model_dir,
        example=val_data[0],
        device=device,
    )

    model.to(device)
    model.eval()
    evaluator = get_evaluator(task, options)
    val_loss, _ = evaluate(
        model,
        device,
        val_loader,
        half_enabled=half_enabled,
        evaluator=evaluator,
    )
    val_scores = evaluator.score()
    val_score = val_scores["score"]
    logger.info(f"Validation score {val_score}.")
    logger.info(f"Validation loss {val_loss}.")
    if task == "point":
        # Log full val set confusion matrix
        evaluator.log_metrics("class0", logger, use_wandb=False)
    if task == "custom":
        # Log full val set MAEs by attribute
        evaluator.log_metrics(logger, use_wandb=False)

    return None


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
