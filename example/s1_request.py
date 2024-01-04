""" Use this script to inference the API with locally stored data


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
