"""VIIRS Vessel Detection Service
"""
from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Response
from pydantic import BaseModel

from src.data.image import prepare_scenes
from src.inference.pipeline import detect_vessels

app = FastAPI()

logger = logging.getLogger(__name__)


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "src", "config", "config.yml"
)

HOST = "0.0.0.0"  # nosec B104
PORT = os.getenv("SVD_PORT", default=5557)

MODEL_VERSION = datetime.today()  # concatenate with git hash


with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["main"]


class SVDResponse(BaseModel):
    """Response object for vessel detections"""

    status: str


class SVDRequest(BaseModel):
    """Request object for vessel detections"""

    scene_id: str  # S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE
    output_dir: str
    raw_path: str
    force_cpu: Optional[bool] = False
    historical1: Optional[str] = None
    historical2: Optional[str] = None
    gcp_bucket: Optional[str] = None
    window_size: Optional[int] = 2048
    padding: Optional[int] = 400
    overlap: Optional[int] = 20
    avoid: Optional[bool] = False
    nms_thresh: Optional[float] = 10
    conf: Optional[float] = 0.9
    save_crops: Optional[bool] = True
    detector_batch_size: int = 4
    postprocessor_batch_size: int = 32
    debug_mode: Optional[bool] = False
    remove_clouds: Optional[bool] = False


@app.on_event("startup")
async def sentinel_init() -> None:
    """Sentinel Vessel Service Initialization"""
    logger.info("Loading model")


async def load_sentinel1_model() -> dict:
    global current_model
    current_model = "sentinel1"
    load_sentinel1_model()


async def load_sentinel2_model() -> dict:
    global current_model
    current_model = "sentinel2"
    load_sentinel2_model()


@app.post("/detections", response_model=SVDResponse)
async def get_detections(info: SVDRequest, response: Response) -> SVDResponse:
    """Returns vessel detections Response object for a given Request object"""
    start = perf_counter()
    scene_id = info.scene_id
    raw_path = info.raw_path
    output = info.output_dir

    with TemporaryDirectory() as tmpdir:
        if not os.path.exists(output):
            logger.debug(f"Creating output dir at {output}")
            os.makedirs(output)

        scratch_path = tmpdir
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not info.force_cpu else "cpu"
        )

        cat_path = os.path.join(scratch_path, scene_id + "_cat.npy")
        base_path = os.path.join(scratch_path, scene_id + "_base.tif")
        catalog = "sentinel" + scene_id[1]  # the second char contains a 1 or 2

        detector_model_dir = config[f"{catalog}_detector"]
        postprocess_model_dir = config[f"{catalog}_postprocessor"]

        if not os.path.exists(cat_path) or not os.path.exists(base_path):
            logger.info("Preprocessing raw scenes.")
            img_array = prepare_scenes(
                raw_path,
                scratch_path,
                scene_id,
                info.historical1,
                info.historical2,
                catalog,
                cat_path,
                base_path,
                device,
                detector_model_dir,
                postprocess_model_dir,
            )

        # Run inference.
        detect_vessels(
            detector_model_dir,
            postprocess_model_dir,
            raw_path,
            scene_id,
            img_array,
            base_path,
            output,
            info.window_size,
            info.padding,
            info.overlap,
            info.conf,
            info.nms_thresh,
            info.save_crops,
            device,
            catalog,
            info.avoid,
            info.remove_clouds,
            info.detector_batch_size,
            info.postprocessor_batch_size,
            debug_mode=info.debug_mode,
        )

        status = str(200)

    elapsed_time = perf_counter() - start
    logger.info(f"SVD {elapsed_time=}, found detections)")

    return SVDResponse(
        status=status,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, proxy_headers=True)
