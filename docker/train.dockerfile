# Build Stage
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04@sha256:9ccfe38d9cb31ae23f6f8f9595b450565da18399c2e71ebc9a5c079f786319e3

SHELL [ "/bin/bash", "-c"]

# Avoid ubuntu prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install some utils
RUN apt-get update && apt-get install software-properties-common -y && \
add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
apt-get update && apt-get install curl git build-essential gdal-bin libgdal-dev libpq-dev python3-gdal python3-dev python3.8-venv python3-pip apt-transport-https ca-certificates gnupg -y


# Install python dependencies
COPY train_requirements.txt /home/vessel_detection/docker/requirements.txt
RUN python3 -m venv .env && source .env/bin/activate && pip install -U pip && pip install -r /home/vessel_detection/docker/requirements.txt

# Cache backbone during build
RUN curl "https://download.pytorch.org/models/resnet50-0676ba61.pth" --create-dirs -o  "/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"

# Copy source code
COPY . /home/vessel_detection
WORKDIR /home/vessel_detection

# Cache backbones during build
RUN curl "https://download.pytorch.org/models/resnet50-0676ba61.pth" --create-dirs -o  "/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
RUN curl "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth" --create-dirs -o "/root/.cache/torch/hub/checkpoints/swin_v2_s-637d8ceb.pth"
RUN curl "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth" --create-dirs -o "/root/.cache/torch/hub/checkpoints/swin_v2_t-b137f0e2.pth"

# WANDB Config
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV RUN_TAG=$RUN_TAG
ENV WANDB_MODE=$WANDB_MODE
ENV WANDB_START_METHOD="thread"
ENV WANDB_PROJECT="vessel-detection"

# Run train script
ENTRYPOINT ["/home/vessel_detection/docker/train_entrypoint.sh"]

# Build:
# From top level of repo:
# export GIT_HASH=$(git rev-parse --short HEAD)
# export IMAGE_TAG=$GIT_HASH
# docker build . -f docker/train.dockerfile -t vessel-detection-train:$IMAGE_TAG

# Run:

# With WANDB logging

# export WANDB_API_KEY=your-api-key

# Train detection model:

# docker run --gpus='"device=0"' --shm-size=16G \
# --rm -v /path/to/local/data:/home/vessel_detection/data \
# -e RUN_TAG=$GIT_HASH \
# -e WANDB_API_KEY=$WANDB_API_KEY \
# -e WANDB_MODE=online \
# vessel-detection-train:$IMAGE_TAG
# --config_path /home/vessel_detection/data/cfg/train_sentinel1_detector.json
# --training_data_dir /home/vessel_detection/data
# --save_dir /home/vessel_detection/data/models
# --metadata_path /home/vessel_detection/data/metadata.sqlite3

# Train attribute prediction model:

# docker run --gpus='"device=0"' --shm-size=16G \
# --rm -v /path/to/local/data:/home/vessel_detection/data \
# -e RUN_TAG=$GIT_HASH \
# -e WANDB_API_KEY=$WANDB_API_KEY \
# -e WANDB_MODE=online \
# vessel-detection-train:$IMAGE_TAG
# --config_path /home/vessel_detection/data/cfg/train_sentinel1_attribute_predictor.json
# --training_data_dir /home/vessel_detection/data
# --save_dir /home/vessel_detection/data/models
# --metadata_path /home/vessel_detection/data/metadata.sqlite3


# or without WANDB logging:

# Train detection model:

# docker run --gpus='"device=0"' --shm-size=16G \
# --rm -v /path/to/local/data:/home/vessel_detection/data \
# -e WANDB_MODE=offline \
# vessel-detection-train:$IMAGE_TAG
# --config_path /home/vessel_detection/data/cfg/train_sentinel1_detector.json
# --training_data_dir /home/vessel_detection/data
# --save_dir /home/vessel_detection/data/models
# --metadata_path /home/vessel_detection/data/metadata.sqlite3

# Train attribute prediction model:

# docker run --gpus='"device=0"' --shm-size=16G \
# --rm -v /path/to/local/data:/home/vessel_detection/data \
# -e WANDB_MODE=offline \
# vessel-detection-train:$IMAGE_TAG
# --config_path /home/vessel_detection/data/cfg/train_sentinel1_attribute_predictor.json
# --training_data_dir /home/vessel_detection/data
# --save_dir /home/vessel_detection/data/models
# --metadata_path /home/vessel_detection/data/metadata.sqlite3