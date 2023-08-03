FROM ubuntu:20.04@sha256:a0a45bd8c6c4acd6967396366f01f2a68f73406327285edc5b7b07cb1cf073db
SHELL [ "/bin/bash", "-c"]

# Avoid ubuntu prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install some utils
RUN apt-get update && apt-get install software-properties-common -y && \
add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
apt-get update && apt-get install curl build-essential gdal-bin unzip libgdal-dev libpq-dev python3-gdal python3.8-venv python3-dev python3-pip -y 

# Install python dependencies
COPY download_requirements.txt /home/vessel_detection/docker/requirements.txt
RUN python3 -m venv .env && source .env/bin/activate && pip install -U pip && pip install -r /home/vessel_detection/docker/requirements.txt

# Main codebase directory
COPY . /home/vessel_detection/
WORKDIR /home/vessel_detection/

# Copernicus API Config
ENV COPERNICUS_USERNAME=$COPERNICUS_USERNAME
ENV COPERNICUS_PASSWORD=$COPERNICUS_PASSWORD

# Set entrypoint
ENTRYPOINT ["/home/vessel_detection/docker/downloader_entrypoint.sh"]

# Build:
# From top level of repo:
# export GIT_HASH=$(git rev-parse --short HEAD)
# export IMAGE_TAG=$GIT_HASH
# docker build . -f docker/downloader.dockerfile -t vessel-detection-download:$IMAGE_TAG

# Run:
# docker run --rm -v /path/to/local/data:/home/vessel_detection/data \
# -e COPERNICUS_USERNAME -e COPERNICUS_PASSWORD \
# vessel-detection-download:$IMAGE_TAG \
# --save_dir="/home/vessel_detection/data/train" \
# --db_path="/home/vessel_detection/data/metadata.sqlite3"