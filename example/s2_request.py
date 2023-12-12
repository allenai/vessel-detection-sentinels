""" Use this script to inference the API with locally stored data

"""
import json
import os
import time

import requests

PORT = os.getenv("SVD_PORT", default=5557)
SVD_ENDPOINT = f"http://localhost:{PORT}/detections"
SAMPLE_INPUT_DIR = "/home/vessel_detection/raw_data/"
SAMPLE_OUTPUT_DIR = "/home/vessel_detection/s2_example/output"
SCENE_ID = "S2A_MSIL1C_20221125T112411_N0400_R037_T30VXK_20221125T132446.SAFE"
TIMEOUT_SECONDS = 600
DEBUG_MODE = True
REMOVE_CLOUDS = True


def sample_request() -> None:
    """Sample request for files stored locally"""
    start = time.time()

    REQUEST_BODY = {
        "raw_path": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "scene_id": SCENE_ID,
        "debug_mode": DEBUG_MODE,
        "remove_clouds": REMOVE_CLOUDS,
    }

    response = requests.post(SVD_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_s2_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    end = time.time()
    print(f"elapsed time: {end-start}")


if __name__ == "__main__":
    sample_request()
