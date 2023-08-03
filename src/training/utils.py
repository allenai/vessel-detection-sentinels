import logging
import typing as t

import mapcalc
import numpy as np
import torch
from google.api_core.exceptions import GoogleAPIError
from google.cloud.storage import Client

from src.training.metric import score


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_iters: int, warmup_factor: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a warmup scheduler for a given optimizer."""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def get_map(target: dict, output: dict) -> float:
    """Compute mean average precision of gt labels and predicted boxes."""
    ground_truth = {
        "boxes": target["boxes"].tolist(),
        "labels": target["labels"].tolist(),
    }
    result_dict = {
        "boxes": output["boxes"].tolist(),
        "labels": output["labels"].tolist(),
        "scores": output["scores"].tolist(),
    }
    return mapcalc.calculate_map(ground_truth, result_dict, 0.5)


def get_score(gt_raw: dict, pred: dict) -> float:
    """Compute best fscore across thresholds ground truth bboxes and associated predictions."""
    gt = {}
    for scene_id, boxes in gt_raw.items():
        if len(boxes) > 0:
            gt[scene_id] = np.array(
                [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes],
                dtype=np.float32,
            )
        else:
            gt[scene_id] = np.zeros((0, 2), dtype=np.float32)

    best_fscore = 0

    for threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cur_pred = {}
        for scene_id, (boxes, scores) in pred.items():
            if len(boxes) > 0:
                cur_pred[scene_id] = np.array(
                    [
                        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        for i, box in enumerate(boxes)
                        if scores[i] >= threshold
                    ],
                    dtype=np.float32,
                )
            else:
                cur_pred[scene_id] = np.zeros((0, 2), dtype=np.float32)

        precision, recall, fscore, _ = score(gt, cur_pred)
        best_fscore = max(best_fscore, fscore)

    return best_fscore


def compute_f1(tp: float, fp: float, fn: float, eps: float = 0.01) -> t.Tuple[float]:
    """Compute F1, precision and recall from true positives, false positives, false negatives.

    Parameters
    ----------
    tp: int
        Number of true positive instances.

    fp: int
        Number of false positive instances.

    fn: int
        Number of false negative instances.

    eps: float
        Lower bound on (precision + recall) necessary to compute F1.

    Returns
    -------
    f1: float
        F1 score.

    precision: float
        Precision score.

    recall: float
        Recall score.
    """
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall < eps:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return (f1, precision, recall)


def upload_file_to_object_storage(
    client: Client,
    bucket: str,
    local_path: str,
    cloud_path: str,
    logger: t.Optional[logging.Logger] = None,
) -> None:
    """Upload a local file to GCS bucket.

    Parameters
    ----------
    client: google.cloud.storage.Client
        Cloud storage client.


    bucket: str
        GCS bucket name.

    local_path: str
        Relative local path to file to upload.

    cloud_path: str
        Absolute object name to be written to in GCS.

    logger: logging.Logger
        Optional logger to which to log exceptions.

    Returns
    -------
    : None
    """
    try:
        bucket = client.get_bucket(bucket)
        target = bucket.blob(cloud_path)
        target.upload_from_filename(local_path)
    except GoogleAPIError as e:

        if logger:
            logger.error(
                f"Failed to upload file at {local_path} to {bucket}/{cloud_path}:\n\n {e}."
            )
        else:
            print(f"Failed to upload file at {local_path} to {bucket}/{cloud_path}.")
    return None
