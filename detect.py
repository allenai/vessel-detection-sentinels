import argparse
import logging
import os
import sys
import typing as t

from src.data.image import prepare_scenes
from src.inference.pipeline import detect_vessels
from src.utils.misc import delete_scratch

import torch

# Configure logger
logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s (%(name)s) (%(levelname)s): %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Detection inference script.")

    parser.add_argument(
        "--raw_path", help="Path where raw scenes are stored", default=None
    )
    parser.add_argument(
        "--scratch_path",
        help="Path to 'scratch space' where warped scenes will be stored.",
    )
    parser.add_argument("--scene_id", help="The current scene to process")
    parser.add_argument(
        "--catalog", help="Catalog (sentinel1 or sentinel2)", default="sentinel1"
    )
    parser.add_argument(
        "--historical1",
        help="Scene ID of first historical image, if available (optional)",
        default=None,
    )
    parser.add_argument(
        "--historical2",
        help="Scene ID of second historical image, if available (optional)",
        default=None,
    )
    parser.add_argument("--output", help="Directory to write output data")
    parser.add_argument(
        "--padding", type=int, help="Padding between sliding window", default=400
    )
    parser.add_argument(
        "--window_size", type=int, help="Inference sliding window size", default=2048
    )
    parser.add_argument(
        "--overlap",
        help="Overlap allowed for predictions between windows",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--detector_model_dir",
        type=str,
        help="Path to dir containing detector model weights and config.",
        default="/home/siv/models/5",
    )
    parser.add_argument(
        "--postprocess_model_dir",
        type=str,
        help="Path to dir containing postprocess model weights and config.",
        default="/home/siv/models/2",
    )
    parser.add_argument(
        "--conf", help="Object detection confidence threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--nms_thresh", help="Run NMS, with this threshold", type=int, default=None
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="If flag is set, inference will use cpu regardless of devices available.",
    )
    parser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help="If flag is set, intermediate processed copies of input imagery used for inference will be retained in specified scratch path.",
    )
    parser.add_argument(
        "--avoid",
        type=str,
        default=None,
        help="Path to a csv specifying locations to avoid in detection outputs.",
    )
    parser.add_argument("--save_crops", type=bool, default=None)

    args = parser.parse_args()
    return args


def main(
    raw_path: str,
    scratch_path: str,
    scene_id: str,
    catalog: str,
    historical1: t.Optional[str],
    historical2: t.Optional[str],
    output: str,
    padding: int,
    window_size: int,
    overlap: int,
    detector_model_dir: str,
    postprocess_model_dir: str,
    nms_thresh: int,
    conf: float,
    force_cpu: bool,
    save_crops: bool,
    keep_intermediate_files: bool,
    avoid: t.Optional[str],
) -> None:
    """Runs inference detection and attribute prediction on specified input."""
    # Create dirs
    if not os.path.exists(scratch_path):
        logger.debug(f"Creating scratch dir at {scratch_path}")
        os.makedirs(scratch_path)
    if not os.path.exists(output):
        logger.debug(f"Creating output dir at {output}")
        os.makedirs(output)

    # Set the inference device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    )

    # Set abspath for avoid locations file
    if avoid:
        avoid = os.path.abspath(avoid)

    # Prepare the warped scenes.
    cat_path = os.path.join(scratch_path, scene_id + "_cat.npy")
    base_path = os.path.join(scratch_path, scene_id + "_base.tif")
    if not os.path.exists(cat_path) or not os.path.exists(base_path):
        logger.info("Preprocessing raw scenes.")
        prepare_scenes(
            raw_path,
            scratch_path,
            scene_id,
            historical1,
            historical2,
            catalog,
            cat_path,
            base_path,
            device,
            detector_model_dir,
            postprocess_model_dir
        )

    # Run inference.
    detect_vessels(
        detector_model_dir,
        postprocess_model_dir,
        raw_path,
        scene_id,
        cat_path,
        base_path,
        output,
        window_size,
        padding,
        overlap,
        conf,
        nms_thresh,
        save_crops,
        device,
        catalog,
        avoid
    )

    # Clean up files added to scratch directory
    if not keep_intermediate_files:
        filenames = os.path.join(
            scratch_path, "*.SAFE*"
        )  # TODO: Return paths from prepare_scenes
        delete_scratch(filenames, scratch_path)


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
