import glob
import json
import logging
import math
import os
import time
import typing as t

from functools import partial

import numpy as np
import pandas as pd
import PIL
import pyproj
import skimage.io
import torch
import torch.utils.data
from osgeo import gdal

from src.data.image import Channels, SUPPORTED_IMAGERY_CATALOGS
from src.models import models
from src.utils.filter import filter_out_locs

PIL.Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("inference")


class GridIndex(object):
    """Implements a grid index for spatial data.

    The index supports inserting points or rectangles, and efficiently searching by bounding box.
    """

    def __init__(self, size):
        self.size = size
        self.grid = {}

    # Insert a point with data.
    def insert(self, p, data):
        self.insert_rect([p[0], p[1], p[0], p[1]], data)

    # Insert a data with rectangle bounds.
    def insert_rect(self, rect, data):
        def f(cell):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(data)

        self.each_cell(rect, f)

    def each_cell(self, rect, f):
        for i in range(rect[0] // self.size, rect[2] // self.size + 1):
            for j in range(rect[1] // self.size, rect[3] // self.size + 1):
                f((i, j))

    def search(self, rect):
        matches = set()

        def f(cell):
            if cell not in self.grid:
                return
            for data in self.grid[cell]:
                matches.add(data)

        self.each_cell(rect, f)
        return matches


def save_detection_crops(
        img: np.ndarray, label: t.NamedTuple, output_dir: str, detect_id: str, catalog: str = "sentinel1",
        out_crop_size: int = 128, transformer: gdal.Transformer = None) -> t.Tuple[np.ndarray, t.List[t.Tuple[float, float]]]:
    """Save detection crops of interest for a given imagery catalog.

    Parameters
    ----------
    img: np.ndarray
        Full preprocessed image (base, no historical concats).

    label: NamedTuple
        Namedtuple with at least preprocess_row and preprocess_column attrs
        specifying row and column in img at which a detection label was made.

    output_dir: str
        Directory in which output crops will be saved.

    detect_id: str
        String identifying detection.

    catalog: str
        String identifying imagery collection. Currently "sentinel1" and
        "sentinel2" are supported.

    out_crop_size: int
        Size of output crop around center of detection.

    transformer: gdal.Transformer
        Transformer specifying source and targer coordinate reference system to use to
        record output crop coordinates (e.g. lat/lon).

    Returns
    -------
    crop: np.ndarray
        A crop with all channels from the preprocessed image.

    corner_lat_lons: list[tuple(float)]
        List of coordinate tuples (lat, lon) of image corners. Ordered as
        upper left, upper right, lower right, lower left, viewed from
        above with north up.
    """
    row, col = label.preprocess_row, label.preprocess_column

    row = np.clip(row, out_crop_size // 2, img.shape[1] - out_crop_size // 2)
    col = np.clip(col, out_crop_size // 2, img.shape[2] - out_crop_size // 2)

    crop_start_row = row - out_crop_size // 2
    crop_end_row = row + out_crop_size // 2
    crop_start_col = col - out_crop_size // 2
    crop_end_col = col + out_crop_size // 2

    crop = img[
        :,
        crop_start_row: crop_end_row,
        crop_start_col: crop_end_col,
    ].numpy()

    # Crop subchannels of interest
    crop_sois = {}
    if catalog == "sentinel1":
        crop_sois["vh"] = crop[0, :, :]
        crop_sois["vv"] = crop[1, :, :]
    elif catalog == "sentinel2":
        crop_sois["tci"] = crop[0:3, :, :].transpose(1, 2, 0)
    else:
        raise ValueError(
            f"You specified imagery catalog={catalog}.\n"
            f"The only supported catalogs are: {SUPPORTED_IMAGERY_CATALOGS}"
        )

    for key, crop_soi in crop_sois.items():
        skimage.io.imsave(
            os.path.join(output_dir, f"{detect_id}_{key}.png"),
            crop_soi,
            check_contrast=False,
        )

    # Get corner coordinates of crops
    corner_lat_lons = []
    corner_cols_and_rows = [
        (crop_start_col, crop_start_row),
        (crop_end_col, crop_start_row),
        (crop_end_col, crop_end_row),
        (crop_start_col, crop_end_row)]
    for corner in corner_cols_and_rows:
        success, point = transformer.TransformPoint(0, float(corner[0]), float(corner[1]), 0)
        if success != 1:
            raise Exception("transform error")
        longitude, latitude = point[0], point[1]
        corner_lat_lons.append((latitude, longitude))

    return crop, corner_lat_lons


def nms(pred: pd.DataFrame, distance_thresh: int = 10) -> pd.DataFrame:
    """Prune detections that are redundant due to a nearby higher-scoring detection.

    Parameters
    ----------
    pred: pd.DataFrame
        Dataframe containing detections, from detect.py.

    distance_threshold: int
        If two detections are closer this threshold, only keep detection
        with a higher score.

    Returns
    -------
    : pd.DataFrame
        Dataframe of detections filtered via NMS.
    """
    # Create table index so we can refer to rows by unique index.
    pred.reset_index()

    elim_inds = set()

    # Create grid index.
    grid_index = GridIndex(max(64, distance_thresh))
    for index, row in enumerate(pred.itertuples()):
        grid_index.insert((row.preprocess_row, row.preprocess_column), index)

    # Remove points that are close to a higher-confidence, not already-eliminated point.
    for index, row in enumerate(pred.itertuples()):
        if row.score == 1:
            continue
        rect = [
            row.preprocess_row - distance_thresh,
            row.preprocess_column - distance_thresh,
            row.preprocess_row + distance_thresh,
            row.preprocess_column + distance_thresh,
        ]
        for other_index in grid_index.search(rect):
            other = pred.loc[other_index]
            if other.score < row.score or (
                other.score == row.score and other_index <= index
            ):
                continue
            if other_index in elim_inds:
                continue

            dx = other.preprocess_column - row.preprocess_column
            dy = other.preprocess_row - row.preprocess_row
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > distance_thresh:
                continue

            elim_inds.add(index)
            break

    logger.info(
        "NMS: retained {} of {} detections.".format(
            len(pred) - len(elim_inds), len(pred)
        )
    )

    return pred.drop(list(elim_inds))


def load_model(model_dir: str, example: list, device: torch.device) -> torch.nn.Module:
    """Load a model from a dir containing config specifying arch, and weights.

    Parameters
    ----------
    model_dir: str
        Directory containing model weights .pth file, and config specifying model
        architechture.

    example: list
        An example input to the model, consisting of two elements. First is
        a torch tensor encoding an image, second is an (optionally None) label.
        TODO: Check second elt is described correctly.

    device: torch.device
        A device on which model should be loaded.

    Returns
    -------
    model: torch.nn.Module
        Loaded model class.
    """
    with open(os.path.join(model_dir, "cfg.json"), "r") as f:
        model_cfg = json.load(f)

    data_cfg = model_cfg["Data"]

    options = model_cfg["Options"]

    channels = Channels(model_cfg["Channels"])
    model_name = model_cfg["Architecture"]

    model_cls = models[model_name]
    model = model_cls(
        {
            "Channels": channels,
            "Device": device,
            "Model": model_cfg,
            "Options": options,
            "Data": data_cfg,
            "Example": example,
        }
    )

    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best.pth"), map_location=device)
    )
    model.to(device)

    model.eval()

    return model


def apply_model(
    detector_dir: str,
    img: np.ndarray,
    window_size: int = 1024,
    padding: int = 0,
    overlap: int = 0,
    threshold: float = 0.5,
    transformer: gdal.Transformer = None,
    nms_thresh: float = None,
    postprocess_model_dir: str = None,
    out_path: str = None,
    catalog: str = 'sentinel1',
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> pd.DataFrame:
    """Apply a model on a large image, by running it on sliding windows along both axes of the image.


    This function currently assumes the task is point detection or the custom attribute prediction task.

    Parameters
    ----------
    detector_dir: str
        Path to dir containing json model config and weights.

    img: np.ndarray
        3D numpy array (channel, row, col). Must be uint8.

    window_size: int
        Size of windows on which to apply the model.

    padding: int

    overlap: int

    threshold: float
        Object detection confidence threshold.

    transformer: gdal.Transformer
        Transformer specifying source and targer coordinate reference system to use to
        record output prediction coordinates (e.g. lat/lon).

    nms_thresh: float
        Distance threshold to use for NMS.

    postprocess_model_dir: str
        Path to dir containing json attribute predictor model config and weights.

    out_path: str
        Path to output directory in which model results will be written.

    device: torch.device
        Device on which model will be applied.

    Returns
    -------
    pred: pd.DataFrame
        Dataframe containing prediction results.
    """
    model = load_model(
        detector_dir,
        example=[img[:, 0:window_size, 0:window_size].float() / 255, None],
        device=device,
    )
    postprocess_model = load_model(
        postprocess_model_dir,
        example=[img[:, 0:128, 0:128].float() / 255, None],
        device=device,
    )
    outputs = []

    with torch.no_grad():
        # Loop over windows.
        row_offsets = (
            [0]
            + list(
                range(
                    window_size - 2 * padding,
                    img.shape[1] - window_size,
                    window_size - 2 * padding,
                )
            )
            + [img.shape[1] - window_size]
        )
        col_offsets = (
            [0]
            + list(
                range(
                    window_size - 2 * padding,
                    img.shape[2] - window_size,
                    window_size - 2 * padding,
                )
            )
            + [img.shape[2] - window_size]
        )

        start_time = time.time()
        for row_offset in row_offsets:
            logger.info(
                f"Processing row offset {row_offset}/{row_offsets[-1]}. {(time.time()-start_time):.1f} seconds elapsed."
            )
            for col_offset in col_offsets:
                crop = img[
                    :,
                    row_offset: row_offset + window_size,
                    col_offset: col_offset + window_size,
                ]
                crop = crop.to(device).float() / 255

                output = model([crop])[0][0]

                # Only keep output detections that are within bounds based
                # on window size, padding, and overlap.
                keep_bounds = [
                    padding,
                    padding,
                    window_size - padding,
                    window_size - padding,
                ]
                if col_offset == 0:
                    keep_bounds[0] = 0
                if row_offset == 0:
                    keep_bounds[1] = 0
                if col_offset >= img.shape[2] - window_size:
                    keep_bounds[2] = window_size
                if row_offset >= img.shape[1] - window_size:
                    keep_bounds[3] = window_size

                keep_bounds[0] -= overlap
                keep_bounds[1] -= overlap
                keep_bounds[2] += overlap
                keep_bounds[3] += overlap

                for pred_idx, box in enumerate(output["boxes"].tolist()):
                    score = output["scores"][pred_idx].item()

                    if score < threshold:
                        continue

                    crop_column = (box[0] + box[2]) / 2
                    crop_row = (box[1] + box[3]) / 2

                    if crop_column < keep_bounds[0] or crop_column > keep_bounds[2]:
                        continue
                    if crop_row < keep_bounds[1] or crop_row > keep_bounds[3]:
                        continue

                    column = col_offset + int(crop_column)
                    row = row_offset + int(crop_row)

                    success, point = transformer.TransformPoint(0, float(column), float(row), 0)
                    if success != 1:
                        raise Exception("transform error")
                    longitude, latitude = point[0], point[1]

                    outputs.append(
                        [
                            row,
                            column,
                            longitude,
                            latitude,
                            score,
                        ]
                    )

        pred = pd.DataFrame(
            data=[output + [0] * 20 for output in outputs],
            columns=[
                "preprocess_row",
                "preprocess_column",
                "lon",
                "lat",
                "score",
                "vessel_length_m",
                "vessel_width_m",
            ]
            + ["heading_bucket_{}".format(i) for i in range(16)]
            + ["vessel_speed_k", "is_fishing_vessel"],
        )
        pred = pred.astype({"preprocess_row": "int64", "preprocess_column": "int64"})
        logger.info("{} detections found".format(len(pred)))

        if nms_thresh is not None:
            pred = nms(pred, distance_thresh=nms_thresh)

        # Post-processing.
        bs = 32
        crop_size = 120
        pred = pred.reset_index(drop=True)
        for x in range(0, len(pred), bs):
            batch_df = pred.iloc[x: min((x + bs), len(pred))]

            crops, indices = [], []
            for idx, b in enumerate(batch_df.itertuples()):
                indices.append(idx)
                row, col = b.preprocess_row, b.preprocess_column

                row = np.clip(row, crop_size // 2, img.shape[1] - crop_size // 2)
                col = np.clip(col, crop_size // 2, img.shape[2] - crop_size // 2)
                if catalog == "sentinel1":
                    crop = img[
                        0:2,
                        row - crop_size // 2: row + crop_size // 2,
                        col - crop_size // 2: col + crop_size // 2,
                    ]
                elif catalog == "sentinel2":
                    crop = img[
                        0:postprocess_model.num_channels,
                        row - crop_size // 2: row + crop_size // 2,
                        col - crop_size // 2: col + crop_size // 2,
                    ]
                else:
                    crop = img[
                        :,
                        row - crop_size // 2: row + crop_size // 2,
                        col - crop_size // 2: col + crop_size // 2,
                    ]
                crop = crop.to(device).float() / 255
                crops.append(crop)

            outputs = postprocess_model(crops)[0].cpu()

            for i in range(len(indices)):
                index = x + i
                pred.loc[index, "vessel_length_m"] = 100 * outputs[i, 0].item()
                pred.loc[index, "vessel_width_m"] = 100 * outputs[i, 1].item()
                heading_probs = torch.nn.functional.softmax(outputs[i, 2:18], dim=0)
                for j in range(16):
                    pred.loc[index, "heading_bucket_{}".format(j)] = heading_probs[
                        j
                    ].item()
                pred.loc[index, "vessel_speed_k"] = outputs[i, 18].item()
                pred.loc[index, "is_fishing_vessel"] = round(
                    torch.nn.functional.softmax(outputs[i, 19:21], dim=0)[1].item(), 15
                )

    return pred


def get_approximate_pixel_size(img: np.ndarray, corner_lat_lons: t.List[t.Tuple[float, float]]) -> t.Tuple[float, float]:
    """Return approximate pixel size given an image (as numpy array), and extremal lat/lons.

    Computes:

    1/2 * [(total width in meters top row) / num_pixels wide + (total width in meters bottom row) / num_pixels wide]

    and

    1/2 * [(total height in meters left col) / num_pixels tall + (total height in meters right col) / num_pixels tall]

    Parameters
    ----------
    img: np.ndarray
        Input numpy array encoding image, shape (C, H, W).

    corner_lat_lons: list
        List of coordinate tuples (lat, lon) of image corners. Ordered as
        upper left, upper right, lower right, lower left, viewed from
        above with north up. Assumed that upper corners share fixed latitude,
        lower corners share fixed latitude, left corners share fixed longitude,
        right corners share fixed longitude.

    Returns
    -------
    approx_pixel_height: float
        Approximate pixel length in image.

    approx_pixel_width: float
        Approximate pixel width in image.
    """
    # Image spatial shape
    n_rows = int(img.shape[1])
    n_cols = int(img.shape[2])

    # Corner coords
    ul, ur, lr, ll = corner_lat_lons

    geodesic = pyproj.Geod(ellps="WGS84")

    _, _, top_width_m = geodesic.inv(ul[1], ul[0], ur[1], ur[0])
    _, _, bottom_width_m = geodesic.inv(ll[1], ll[0], lr[1], lr[0])

    approx_pixel_width = .5 * ((top_width_m // n_cols) + (bottom_width_m // n_cols))
    _, _, left_height_m = geodesic.inv(ul[1], ul[0], ll[1], ll[0])
    _, _, right_height_m = geodesic.inv(ur[1], ur[0], lr[1], lr[0])

    approx_pixel_height = .5 * ((left_height_m // n_rows) + (right_height_m // n_rows))

    return approx_pixel_height, approx_pixel_width


def detect_vessels(
    detector_model_dir: str,
    postprocess_model_dir: str,
    raw_path: str,
    scene_id: str,
    cat_path: str,
    base_path: str,
    output_dir: str,
    window_size: int,
    padding: int,
    overlap: int,
    conf: float,
    nms_thresh: float,
    save_crops: bool,
    device: torch.device,
    catalog: str,
    avoid: t.Optional[str]
) -> None:
    """Detect vessels in specified image using specified model.

    Parameters
    ----------
    detector_model_dir: str
        Path to dir containing json model config and weights.

    postprocess_model_dir: str
        Path to dir containing json attribute predictor model config and weights.

    raw_path: str
        Path to dir containing images on which inference will be performed.

    scene_id: str
        The directory name of a (decompressed) Sentinel-1 scene in the raw_path
        directory on which inference will be performed.

        E.g. S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE

    cat_path: str
        The path to an intermediate numpy array containing a preprocessed target
        consisting of concatenated inference target and optional historical imagery.
        See detect.py for an example of usage.

    base_path: str
        The path to a preprocessed copy of the inference target geotiff file.
        See detect.py for an example of usage.

    output_dir: str
        Path to output directory in which model results will be written.

    window_size: int
        Size of windows on which to apply the model.

    padding: int

    overlap: int

    conf: float
        Object detection confidence threshold.

    nms_thresh: float
        Distance threshold to use for NMS.

    save_crops: bool
        If True, crops surrounding point detections will be saved to output dir.

    device: torch.device
        Device on which model will be applied.

    catalog: str
        Imagery catalog. Currently supported: "sentinel1", "sentinel2".

    avoid: Optional[str]
        If not None, a path to a csv file containing columns lon, lat, width_m.
        Every row should have lon and lat specified, and optionally width_m specified.
        Locations specified will be used to filter, using a default extent (or width_m if specified),
        any detections that overlap. Could be used to filter out e.g. known fixed infrastructure.
    """
    # Isolate original file name to write with detections
    suffix = "_cat.npy"
    if cat_path.endswith(suffix):
        filename = cat_path.split("/")[-1][: -len(suffix)]
    else:
        filename = "scene"

    # For greyscale input, fix image to have three dimensions.
    if cat_path.endswith(".npy"):
        img = np.load(cat_path)
    else:
        img = skimage.io.imread(cat_path)
        if len(img.shape) == 2:
            img = img[None, :, :]
        else:
            img = img.transpose(2, 0, 1)
    img = torch.as_tensor(img)

    layer = gdal.Open(base_path)
    transformer = gdal.Transformer(layer, None, ["DST_SRS=WGS84"])

    pred = apply_model(
        detector_model_dir,
        img,
        window_size=window_size,
        padding=padding,
        overlap=overlap,
        threshold=conf,
        transformer=transformer,
        nms_thresh=nms_thresh,
        postprocess_model_dir=postprocess_model_dir,
        out_path=output_dir,
        catalog=catalog,
        device=device,
    )

    # Add pixel coordinates of detections w.r.t. original image
    # TODO: Modularize this, and make it robust to different sources, etc.
    def get_input_pixel_coords(out_col, out_row, transformer):
        success, point = transformer.TransformPoint(
            0, float(out_col), float(out_row), 0
        )
        input_col, input_row, _ = point
        return int(input_col), int(input_row)

    if catalog == "sentinel1":
        measurement_path = os.path.join(raw_path, scene_id, "measurement")
        source_paths = os.listdir(measurement_path)
        input_path = os.path.join(measurement_path, source_paths[0])
    elif catalog == "sentinel2":
        base_channel = "TCI"
        raw_match = f"GRANULE/*/IMG_DATA/*_{base_channel}.jp2"
        path_pattern = os.path.join(raw_path, scene_id, raw_match)
        paths = glob.glob(path_pattern)
        input_path = paths[0]

    else:
        raise ValueError(
            f"You specified imagery catalog={catalog}.\n"
            f"The only supported catalogs are: {SUPPORTED_IMAGERY_CATALOGS}"
        )

    output_raster = gdal.Open(base_path)
    input_raster = gdal.Open(input_path)
    transformer = gdal.Transformer(output_raster, input_raster, [])
    get_input_coords = partial(get_input_pixel_coords, transformer=transformer)
    del output_raster, input_raster
    if len(pred) > 0:
        pred[["column", "row"]] = pred.apply(
            lambda row: get_input_coords(row.preprocess_column, row.preprocess_row),
            axis=1,
            result_type="expand",
        )

    pred = pred.reset_index(drop=True)

    # Construct scene and detection ids, crops if requested
    transformer = gdal.Transformer(layer, None, ["DST_SRS=WGS84"])
    detect_ids = [None] * len(pred)
    scene_ids = [None] * len(pred)
    for index, label in enumerate(pred.itertuples()):
        scene_id = filename
        scene_ids[index] = scene_id
        detect_id = "{}_{}".format(filename, index)
        detect_ids[index] = detect_id

        if save_crops:
            transformer = gdal.Transformer(layer, None, ["DST_SRS=WGS84"])
            crop, corner_lat_lons = save_detection_crops(
                img, label, output_dir, detect_id, catalog=catalog, out_crop_size=128, transformer=transformer)

            # CW rotation angle necessary to rotate vertical line in image to align with North
            pred["orientation"] = 0  # by virtue of crops coming from web mercator aligned image.

            # TODO: This approach assumes all crops associated with a detection have the same pixel
            # resolution, which need not always be the case.
            # Avoid storing crop metadata on the detection object directly.
            _, pixel_width = get_approximate_pixel_size(crop, corner_lat_lons)
            pred["meters_per_pixel"] = pixel_width

    # Insert scene/detect ids in csv
    pred.insert(len(pred.columns), "detect_id", detect_ids)
    pred.insert(len(pred.columns), "scene_id", scene_ids)

    # Filter out undesirable locations
    if avoid:
        logger.info(f"Filtering detections based on locs in {avoid}.")
        num_unfiltered = len(pred)
        pred = filter_out_locs(pred, loc_path=avoid)
        logger.info(f"Retained {len(pred)} of {num_unfiltered} detections.")

    pred.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    return None
