""" Use this script to inference the API with locally stored data

This is the docker command that was used before the API was created:

docker run --shm-size 16G --gpus='"device=0"' \
-v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
--raw_path=/home/vessel_detection/data/ \
--scratch_path=/home/vessel_detection/data/scratch/ \
--output=/home/vessel_detection/data/output/ \
--detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/3dff445 \
--postprocess_model_dir=/home/vessel_detection/data/models/attr/c34aa37 \
--historical1=S1B_IW_GRDH_1SDV_20211118T025212_20211118T025237_029636_03896D_FCBE.SAFE \
--historical2=S1B_IW_GRDH_1SDV_20211106T025212_20211106T025237_029461_03840B_6E73.SAFE \
--scene_id=S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE \
--conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel1

"""
import json
import os
import time

import requests

PORT = os.getenv("SVD_PORT", default=5557)
SVD_ENDPOINT = f"http://localhost:{PORT}/detections"
SAMPLE_INPUT_DIR = "/home/vessel_detection/raw_data/"
SAMPLE_OUTPUT_DIR = "/home/vessel_detection/s1_example/output"
SCENE_ID = "S1A_IW_GRDH_1SDV_20221002T205126_20221002T205156_045268_056950_0664.SAFE"
HISTORICAL_1 = (
    "S1A_IW_GRDH_1SDV_20221014T205126_20221014T205155_045443_056F2D_861B.SAFE"
)
HISTORICAL_2 = (
    "S1A_IW_GRDH_1SDV_20221026T205127_20221026T205156_045618_05745F_B918.SAFE"
)
TIMEOUT_SECONDS = 600


def sample_request() -> None:
    """Sample request for files stored locally"""
    start = time.time()

    REQUEST_BODY = {
        "raw_path": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "scene_id": SCENE_ID,
        "historical_1": HISTORICAL_1,
        "historical_2": HISTORICAL_2,
    }

    response = requests.post(SVD_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_s1_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    end = time.time()
    print(f"elapsed time: {end-start}")


if __name__ == "__main__":
    sample_request()
