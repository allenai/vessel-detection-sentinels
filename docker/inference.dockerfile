FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04@sha256:9ccfe38d9cb31ae23f6f8f9595b450565da18399c2e71ebc9a5c079f786319e3
SHELL [ "/bin/bash", "-c"]

# Avoid ubuntu prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install some utils
RUN apt-get update && apt-get install software-properties-common -y && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    apt-get update && apt-get install curl build-essential gdal-bin libgdal-dev libpq-dev python3-gdal python3.8-venv python3-dev python3-pip -y

# Install python dependencies
COPY inference_requirements.txt /home/vessel_detection/docker/requirements.txt
RUN python3 -m venv .env && source .env/bin/activate && pip install -U pip && pip install -r /home/vessel_detection/docker/requirements.txt

# Main codebase directory
COPY . /home/vessel_detection/
WORKDIR /home/vessel_detection/

# Cache backbones during build
RUN curl "https://download.pytorch.org/models/resnet50-0676ba61.pth" --create-dirs -o  "/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
RUN curl "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth" --create-dirs -o "/root/.cache/torch/hub/checkpoints/swin_v2_s-637d8ceb.pth"
RUN curl "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth" --create-dirs -o "/root/.cache/torch/hub/checkpoints/swin_v2_t-b137f0e2.pth"

# Set entrypoint
ENTRYPOINT ["/home/vessel_detection/docker/inference_entrypoint.sh"]

# Build:
# From top level of repo:
# export IMAGE_TAG=$(git rev-parse HEAD)
# docker build . -f docker/inference.dockerfile -t vessel-detection:$IMAGE_TAG

# Run:
# docker run --shm-size 16G --gpus='"device=0"' \
# -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
# --raw_path=/home/vessel_detection/data/ \
# --scratch_path=/home/vessel_detection/scratch/ \
# --output=/home/vessel_detection/output/ \
# --detector_model_dir=/home/vessel_detection/models/5 \
# --postprocess_model_dir=/home/vessel_detection/models/2 \
# --scene_id=S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE \
# --historical1=S1B_IW_GRDH_1SDV_20211118T025212_20211118T025237_029636_03896D_FCBE.SAFE \
# --historical2=S1B_IW_GRDH_1SDV_20211106T025212_20211106T025237_029461_03840B_6E73.SAFE \
# --conf=.9 --nms_thresh=10 --save_crops=True