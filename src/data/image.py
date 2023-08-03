import datetime
import glob
import json
import logging
import multiprocessing
import os
import sys
import typing as t

import numpy as np
import skimage.io
import torch

from src.data.retrieve import RetrieveImage
from src.data.warp import warp

# Configure logger
logger = logging.getLogger("prepare_scenes")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s (%(levelname)s): %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

SENTINEL_1_REQUIRED_CHANNEL_TYPES = ["vh", "vv"]

# Map Sentinel-2 channels according to DB naming convention
# to raw imagery path naming convention, and channel count.
# Unlike Sentinel-1, actual channels used by model are specified
# via config that can be altered at runtime.
SENTINEL_2_CHANNEL_MAPPING = {"tci": {"path_abbrev": "TCI", "count": 3}, "b08": {"path_abbrev": "B08", "count": 1},
                              "b11": {"path_abbrev": "B11", "count": 1}, "b12": {"path_abbrev": "B12", "count": 1}}
SUPPORTED_IMAGERY_CATALOGS = ["sentinel1", "sentinel2"]


class InvalidDataError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class Channels(object):
    def __init__(self, channels):
        """
        Given a (JSON-decoded) list of fChannel, creates a Channels object.
        """
        self.channels = channels

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        return self.channels[idx]

    def count(self):
        return sum([channel["Count"] for channel in self.channels])

    def flatten(self):
        """
        For example, [tci, b5] -> ['tci-0', 'tci-1', 'tci-2', 'b5']
        """
        flat_list = []
        for channel in self.channels:
            if channel["Count"] > 1:
                for i in range(channel["Count"]):
                    flat_list.append("{}-{}".format(channel["Name"], i))
            else:
                flat_list.append(channel["Name"])
        return flat_list

    def with_ranges(self):
        l = []
        cur_idx = 0
        for channel in self.channels:
            l.append((channel, (cur_idx, cur_idx + channel["Count"])))
            cur_idx += channel["Count"]
        return l


def warp_func(job: list):
    """Perform a single image warp job.

    Parameters
    ----------
    job: list
        List specifying job params.

    Returns
    -------
    : ImageInfo
        Class containing warped image metadata.
    """
    retrieve_image, scene_id, channel_name, src_path, dst_path = job
    return warp(retrieve_image, src_path, dst_path, projection="epsg:3857")


def prepare_scenes(
    raw_path: str,
    scratch_path: str,
    scene_id: str,
    historical1: t.Optional[str],
    historical2: t.Optional[str],
    catalog: str,
    cat_path: str,
    base_path: str,
    device: torch.device,
    detector_model_dir: str,
    postprocess_model_dir: str
) -> None:
    """Extract and warp scenes, then save numpy array with scene info.

    Parameters
    ----------
    raw_path: str
        Path to directory containing raw inference image and (optionally) historical overlaps.

    scratch_path: str
        Path to directory where intermediate files can be populated.

    scene_id: str
        File name of raw inference image target in raw_path dir.

    historical1: Optional[str]
        File name of raw inference image overlap-1 in raw_path dir.

    historical2: Optional[str]
        File name of raw inference image overlap-1 in raw_path dir.

    catalog: str
        Imagery catalog. For now, only Sentinel-1 ("sentinel1") and
        Sentinel-2 ("sentinel2") are covered.

    cat_path: str
        Path to pre-processed numpy array containing preprocessed, concatenated overlaps.

    base_path: str
        Path to preprocessed copy of original inference target tif image.

    device: torch.device
        Device to use to prepare scenes.

    detector_model_dir: str
        Path to dir containing json model config and weights.

    postprocess_model_dir: str
        Path to dir containing json attribute predictor model config and weights.

    Returns
    -------
    : None

    """
    # Load detector and postprocessor config
    with open(os.path.join(detector_model_dir, "cfg.json"), "r") as f:
        detector_cfg = json.load(f)
    with open(os.path.join(postprocess_model_dir, "cfg.json"), "r") as f:
        postprocess_cfg = json.load(f)

    # Verify channel requirements align (ignoring overlap channels),
    # since the models will share the same pre-processed base imagery
    postprocess_channels = set(ch["Name"] for ch in postprocess_cfg["Channels"] if "overlap" not in ch["Name"])
    detector_channels = set(ch["Name"] for ch in detector_cfg["Channels"] if "overlap" not in ch["Name"])
    if postprocess_channels != detector_channels:
        raise InvalidConfigError("Detector and postprocessor models are required to use the"
                                 f"same underlying channels.\n You passed"
                                 f" detector_channels={detector_channels}\n"
                                 f"postprocessor_channels={postprocess_channels}")

    # Warp the scenes, in parallel.
    # Each job is (retrieve_image, scene_id, channel_name, src_path, dst_path)
    retrieve_images = []
    jobs = []
    scene_ids = [scene_id]
    if historical1:
        scene_ids.append(historical1)
    if historical2:
        scene_ids.append(historical2)
    for scene_id in scene_ids:
        scene_channels = []
        if catalog == "sentinel1":
            measurement_path = os.path.join(raw_path, scene_id, "measurement")
            fnames = {
                fname.split("-")[3]: fname for fname in os.listdir(measurement_path)
            }
            if all(key in fnames for key in SENTINEL_1_REQUIRED_CHANNEL_TYPES):
                scene_channels.append(
                    {
                        "Name": "vh",
                        "Path": os.path.join(measurement_path, fnames["vh"]),
                        "Count": 1,
                    }
                )
                scene_channels.append(
                    {
                        "Name": "vv",
                        "Path": os.path.join(measurement_path, fnames["vv"]),
                        "Count": 1,
                    }
                )
            else:
                raise InvalidDataError(
                    f"Raw Sentinel-1 data must contain polarization channels={SENTINEL_1_REQUIRED_CHANNEL_TYPES}.\n"
                    f"Found: {fnames}"
                )
        elif catalog == "sentinel2":
            # Channels of interest for model, as specified in model cfg
            # Length rules out overlap channels from cfg, which are handled separately here
            sentinel_2_cois = [x["Name"] for x in detector_cfg["Channels"] if len(x["Name"]) == 3]
            sentinel_2_coi_map = dict((k, SENTINEL_2_CHANNEL_MAPPING[k])
                                      for k in sentinel_2_cois if k in SENTINEL_2_CHANNEL_MAPPING)
            for channel, val in sentinel_2_coi_map.items():
                path_abbrev = val["path_abbrev"]
                count = val["count"]
                path_pattern = os.path.join(raw_path, scene_id, f"GRANULE/*/IMG_DATA/*_{path_abbrev}.jp2")
                paths = glob.glob(path_pattern)
                if len(paths) == 1:
                    path = paths[0]
                    scene_channels.append(
                        {
                            "Name": channel,
                            "Path": path,
                            "Count": count
                        }
                    )
                else:
                    raise InvalidDataError(
                        f"Raw Sentinel-2 data must be of L1C product type, and contain channel={channel}.\n"
                        f"Did not find a unique path using the pattern: {path_pattern}"
                    )

        else:
            raise ValueError(
                f"You specified imagery catalog={catalog}.\n"
                f"The only supported catalogs are: {SUPPORTED_IMAGERY_CATALOGS}"
            )

        retrieve_image = RetrieveImage(
            uuid="x",
            name=scene_id,
            time=datetime.datetime.now(),
            format="x",
            channels=scene_channels,
            pixel_size=10,
        )
        retrieve_image.job_ids = []

        for ch in scene_channels:
            retrieve_image.job_ids.append(len(jobs))

            if len(jobs) == 0:
                dst_path = base_path
            else:
                dst_path = os.path.join(
                    scratch_path, scene_id + "_" + ch["Name"] + ".tif"
                )

            jobs.append(
                [
                    retrieve_image,
                    scene_id,
                    ch["Name"],
                    ch["Path"],
                    dst_path,
                ]
            )

        retrieve_images.append(retrieve_image)

    p = multiprocessing.Pool(8)
    image_infos = p.map(warp_func, jobs)
    p.close()

    first_info = None
    ims = []
    for retrieve_image in retrieve_images:
        overlap_offset = (0, 0)
        for ch_idx, ch in enumerate(retrieve_image.channels):
            job_id = retrieve_image.job_ids[ch_idx]
            job = jobs[job_id]
            image_info = image_infos[job_id]
            _, _, _, _, tmp_path = job

            im = skimage.io.imread(tmp_path)
            im = np.clip(im, 0, 255).astype(np.uint8)
            if len(im.shape) == 2:
                im = im[None, :, :]
            else:
                im = im.transpose(2, 0, 1)
            im = torch.as_tensor(im)

            if not first_info:
                first_info = image_info
                ims.append(im)
                continue

            # Align later images with the first one.
            left = image_info.column - first_info.column
            top = image_info.row - first_info.row

            # Automatically fix mis-alignment in the georeference metadata.
            # We re-align by maximizing the dot product between the overlapping image with small offsets, and the base image.
            if "vh" in ch["Name"] or "tci" in ch["Name"]:
                base_padded = ims[0][0, :, :]
                other_padded = im[0, :, :]
                if top < 0:
                    other_padded = other_padded[-top:, :]
                else:
                    base_padded = base_padded[top:, :]
                if left < 0:
                    other_padded = other_padded[:, -left:]
                else:
                    base_padded = base_padded[:, left:]

                if other_padded.shape[0] > base_padded.shape[0]:
                    other_padded = other_padded[0: base_padded.shape[0], :]
                else:
                    base_padded = base_padded[0: other_padded.shape[0], :]
                if other_padded.shape[1] > base_padded.shape[1]:
                    other_padded = other_padded[:, 0: base_padded.shape[1]]
                else:
                    base_padded = base_padded[:, 0: other_padded.shape[1]]

                # Try re-alignments up to 32 pixels off from the georeference metadata.
                max_offset_orig = 32
                # Test dot products at 1/4 the original resolution.
                realign_scale = 4

                base_padded = (
                    base_padded[::realign_scale, ::realign_scale].float().to(device)
                )
                other_padded = (
                    other_padded[::realign_scale, ::realign_scale].float().to(device)
                )

                max_offset = max_offset_orig // realign_scale
                best_offset = None
                best_score = None
                for top_offset in range(-max_offset, max_offset):
                    for left_offset in range(-max_offset, max_offset):
                        # Crop the base by a constant amount, i.e., the maximum offset.
                        # Then crop the other image by anywhere between 0 and 2*max_offset on the first side, and the opposite on the other side.
                        cur_base = base_padded[
                            max_offset:-max_offset, max_offset:-max_offset
                        ]
                        cur_other = other_padded[
                            max_offset
                            - top_offset: other_padded.shape[0]
                            - (max_offset + top_offset),
                            max_offset
                            - left_offset: other_padded.shape[1]
                            - (max_offset + left_offset),
                        ]
                        score = torch.mean(cur_base * cur_other)
                        if best_score is None or score > best_score:
                            best_offset = (left_offset, top_offset)
                            best_score = score

                overlap_offset = (
                    realign_scale * best_offset[0],
                    realign_scale * best_offset[1],
                )
                logger.info(
                    "computed best offset for {}: {}".format(
                        retrieve_image.name, overlap_offset
                    )
                )

            left += overlap_offset[0]
            top += overlap_offset[1]

            if top < 0:
                im = im[:, -top:, :]
            else:
                im = torch.nn.functional.pad(im, (0, 0, top, 0))
            if left < 0:
                im = im[:, :, -left:]
            else:
                im = torch.nn.functional.pad(im, (left, 0, 0, 0))

            # Crop to size if needed.
            if im.shape[1] > first_info.height:
                im = im[:, 0: first_info.height, :]
            elif im.shape[1] < first_info.height:
                im = torch.nn.functional.pad(
                    im, (0, 0, 0, first_info.height - im.shape[1])
                )
            if im.shape[2] > first_info.width:
                im = im[:, :, 0: first_info.width]
            elif im.shape[2] < first_info.width:
                im = torch.nn.functional.pad(
                    im, (0, first_info.width - im.shape[2], 0, 0)
                )

            ims.append(im)

    im = torch.cat(ims, dim=0)
    logger.debug(f"Writing numpy array at {cat_path}")
    np.save(cat_path, im.numpy())

    return None
