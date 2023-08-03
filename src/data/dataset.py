import json
import random

import numpy as np
import skimage.io
import torch
import torchvision

from src.data.tiles import load_window

# Weight given to positive fishing activity class instances
# relative to negative fishing activity class instances
# TODO: Pass or compute at runtime.
NEGATIVE_VESSEL_TYPE_FACTOR = 32

# Heuristic based on qualitative PCA performance.
PSEUDO_HEADING_THRESHOLD_M = 100

# Functions to convert labels at a window into a target dict.


def detect_preprocess(
    task: str,
    window: dict,
    target_width: int,
    target_height: int,
    col_factor: int,
    row_factor: int,
) -> dict:
    """Construct target dict for detection task from window labels.

    Parameters
    ----------
    task: str
        Task label (currently point is supported).

    window: dict
        Window as in sqlite metadata database.

    target_width:

    target_height:

    col_factor: int

    row_factor: int

    Returns
    -------
    : dict
        Target dictionary with which to train a Faster RCNN model.
    """
    centers = []
    boxes = []
    class_labels = []

    for label in window["labels"]:
        # Throwout labels produced to facilitate annotation work
        if label["properties"] and "OnKey" in label["properties"]:
            continue

        if task == "point":
            column = int((label["column"] - window["column"]) * col_factor)
            row = int((label["row"] - window["row"]) * row_factor)

            xmin = column - 20
            xmax = column + 20
            ymin = row - 20
            ymax = row + 20
        else:
            xmin = label["column"] - window["column"]
            xmax = xmin + label["width"]
            ymin = label["row"] - window["row"]
            ymax = ymin + label["height"]
            column = (xmin + xmax) // 2
            row = (ymin + ymax) // 2

        centers.append([column, row])
        boxes.append([xmin, ymin, xmax, ymax])
        class_labels.append(0)

    if len(boxes) == 0:
        centers = torch.zeros((0, 2), dtype=torch.float32)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.zeros((0,), dtype=torch.int64)
    else:
        centers = torch.as_tensor(centers, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_labels = torch.as_tensor(class_labels, dtype=torch.int64)

    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # Create target dictionary in expected format for Faster R-CNN
    return {
        "centers": centers,
        "boxes": boxes,
        "labels": class_labels,
        "area": area,
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
    }


def pixel_regression_preprocess(
    task: str,
    window: dict,
    target_width: int,
    target_height: int,
    col_factor: int,
    row_factor: int,
) -> dict:
    """Construct target dict for pixel level regression task from window labels.

    Parameters
    ----------
    task: str
        Task label.

    window: dict
        Window as in sqlite metadata database.

    target_width: int

    target_height: int

    col_factor: int

    row_factor: int

    Returns
    -------
    : dict
        Target dictionary with which to train a pixel level regression model.
    """
    target_img = np.zeros((target_height, target_width), dtype=np.float32)

    labels = window["labels"]
    labels.sort(key=lambda label: label["value"])

    for label in labels:
        poly_coords = json.loads(label["extent"])
        col_coords = [
            (coord[0] - window["column"]) * col_factor for coord in poly_coords
        ]
        row_coords = [(coord[1] - window["row"]) * row_factor for coord in poly_coords]
        rr, cc = skimage.draw.polygon(row_coords, col_coords, shape=target_img.shape)
        target_img[rr, cc] = label["value"]

    return {"target": torch.as_tensor(target_img)}


def regression_preprocess(
    task: str,
    window: dict,
    target_width: int,
    target_height: int,
    col_factor: int,
    row_factor: int,
):
    """Construct target dict for general regression task from window labels.

    Parameters
    ----------
    task: str
        Task label.

    window: dict
        Window as in sqlite metadata database.

    target_width: int

    target_height: int

    col_factor: int

    row_factor: int

    Returns
    -------
    : dict
        Target dictionary with which to train a regression model.
    """
    if len(window["labels"]) > 0:
        label = window["labels"][0]
        value = label["value"]
    else:
        value = 0

    return {"target": torch.tensor(value, dtype=torch.float32)}


def custom_preprocess(
    task: str,
    window: dict,
    target_width: int,
    target_height: int,
    col_factor: int,
    row_factor: int,
) -> dict:
    """Construct target dict for custom attribute prediction task from window labels.

    Parameters
    ----------
    task: str
        Task label.

    window: dict
        Window as in sqlite metadata database.

    target_width: int

    target_height: int

    col_factor: int

    row_factor: int

    Returns
    -------
    : dict
        Target dictionary with which to train a custom attribute prediction model.
    """
    # One label per window in this context
    label = window["labels"][0]
    properties = json.loads(label["properties"])
    target = torch.zeros((21,), dtype=torch.float32)
    target[0] = properties["Length"] / 100
    target[1] = properties["Width"] / 100

    if properties["Length"] > PSEUDO_HEADING_THRESHOLD_M and properties.get("PseudoHeading", None):
        heading_bucket = properties["PseudoHeading"] * 16 // 360
    else:
        heading_bucket = properties["Heading"] * 16 // 360

    buckets = 16
    target[2 + heading_bucket] = 8 / 14
    # Give weight to adjacent heading buckets
    # TODO: Only assign this weight if label is near bin boundary
    for offset in [-1, 1]:
        cur = (heading_bucket + 16 + offset) % buckets
        target[2 + cur] = 1 / 14

    # Give equal weight to opposite heading bucket, as pseudo labels are
    # based on undirected alignment.
    opp_bucket = int((heading_bucket + buckets/2) % buckets)
    target[2 + opp_bucket] = 2 / 14
    for offset in [-1, 1]:
        cur = (opp_bucket + offset) % buckets
        target[2 + cur] = 1 / 14

    target[18] = properties["Speed"]
    vessel_type = properties["ShipAndCargoType"]
    fishing = vessel_type == 30
    # not_available = vessel_type == 0
    if fishing:
        target[19] = 0.0
        target[20] = 1.0
    # elif not_available:
    #     target[19] = 0.5
    #     target[20] = 0.5
    else:
        target[19] = 1.0
        target[20] = 0.0

    return {"target": target}


# Map from task type to preprocess function.
label_preprocessors = {
    "point": detect_preprocess,
    "box": detect_preprocess,
    "pixel_regression": pixel_regression_preprocess,
    "regression": regression_preprocess,
    "custom": custom_preprocess,
}


class Dataset(object):
    """
    Pytorch dataset for working with Sentinel-1 data.
    """

    def __init__(
        self,
        dataset,
        windows,
        channels,
        splits=None,
        transforms=None,
        image_size=None,
        chip_size=None,
        valid=False,
        preprocess_dir="../data/preprocess",
    ):

        self.dataset = dataset
        self.channels = channels
        self.transforms = transforms
        self.image_size = image_size
        self.chip_size = chip_size
        self.valid = valid
        self.preprocess_dir = preprocess_dir

        self.windows = windows
        if splits:
            splits = set(splits)
            self.windows = [
                window for window in self.windows if window["split"] in splits
            ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        if self.chip_size:
            if window["Width"] < self.chip_size:
                pick_col = 0
            elif self.valid:
                pick_col = window["Width"] // 2 - self.chip_size // 2
            else:
                pick_col = random.randint(0, window["width"] - self.chip_size)

            if window["height"] < self.chip_size:
                pick_row = 0
            elif self.valid:
                pick_row = window["height"] // 2 - self.chip_size // 2
            else:
                pick_row = random.randint(0, window["height"] - self.chip_size)

            window = {
                "ID": window["id"],
                "Image": window["image"],
                "Labels": window["labels"],
                "Width": self.chip_size,
                "Height": self.chip_size,
                "Column": window["column"] + pick_col,
                "Row": window["row"] + pick_row,
            }

        if self.image_size:
            target_width, target_height = self.image_size, self.image_size
            col_factor = target_width / window["width"]
            row_factor = target_height / window["height"]
        else:
            target_width = window["width"]
            target_height = window["height"]
            col_factor = 1
            row_factor = 1

        data = torch.zeros(
            (self.channels.count(), target_height, target_width), dtype=torch.uint8
        )

        for channel, rng in self.channels.with_ranges():
            im = load_window(
                window["image"]["uuid"],
                channel,
                window["column"],
                window["row"],
                window["width"],
                window["height"],
                preprocess_dir=self.preprocess_dir,
            )
            im = torch.as_tensor(im)

            if channel["Count"] > 1:
                im = im.permute(2, 0, 1)

            if im.shape[0] != target_height or im.shape[1] != target_width:
                im = torchvision.transforms.functional.resize(
                    im.unsqueeze(0), [target_height, target_width]
                ).squeeze(0)

            data[rng[0]: rng[1], :, :] = im

        task = self.dataset["task"]

        target = {
            "chip": (window["image"]["name"], window["column"], window["row"]),
            "window_id": torch.tensor(window["id"]),
            "image_id": torch.tensor(idx),
        }

        preprocess_func = label_preprocessors[task]
        target.update(
            preprocess_func(
                task, window, target_width, target_height, col_factor, row_factor
            )
        )

        if self.transforms is not None:
            data, target = self.transforms(data, target)

        return data, target

    def get_bg_balanced_sampler(self, background_frac=1.0):
        """
        Returns a torch.utils.data.Sampler that samples foreground/background at 1:1 ratio.
        Background is defined as having no labels.
        """
        bg_indices = []
        fg_indices = []
        for i, window in enumerate(self.windows):
            if len(window["labels"]) >= 1:
                fg_indices.append(i)
            else:
                bg_indices.append(i)

        bg_weight = background_frac * (len(fg_indices) / len(bg_indices))
        weights = [0.0] * len(self.windows)
        for i in fg_indices:
            weights[i] = 1.0
        for i in bg_indices:
            weights[i] = bg_weight

        print(
            "using bg_balanced_sampler with {} bg chips, {} fg chips (bg_weight={})".format(
                len(bg_indices), len(fg_indices), bg_weight
            )
        )
        return torch.utils.data.WeightedRandomSampler(weights, len(self.windows))
