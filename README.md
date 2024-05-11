# Sentinel-1 and Sentinel-2 Vessel Detection

This repository provides tools that can be used to detect vessels in
synthetic aperture radar (SAR) imagery produced by the Sentinel-1 satellite constellation, and electro-optical (EO)
and infrared (IR) imagery produced by the Sentinel-2 satellite constellation. See: 
[Satellite Imagery and AI: A New Era in Ocean Conservation, from Research to Deployment and Impact](https://arxiv.org/abs/2312.03207)

These models run at global scale in production in [Skylight](https://www.skylight.global/), a free maritime intellignece platform that supports ocean conservation efforts worldwide.  

## **Dataset**

The annotations are included in data/metadata.sqlite3 with the following schema. The raw data are intended to be downloaded from source (ESA), via the included download scipts. 

### Metadata Schema

The metadata we provide is organized according to a schema spanning a few tables. Toggle the below to read more about the details of these tables.

<details>
<summary><b>Schema</b> </summary>
The sqlite database has the following tables (and associated schemas):

1. collections

   A collection specifies a source or catalog of imagery.

   **Examples**:

   Sentinel-1, Sentinel-2, or Landsat-8.

   **Schema**:

   | cid | name | type    | notnull | dflt_value | pk  |
   | --- | ---- | ------- | ------- | ---------- | --- |
   | 0   | id   | INTEGER | 0       |            | 1   |
   | 1   | name | TEXT    | 0       |            | 0   |

   **Example Row**:
   |id| name|
   |---|----|
   |1|sentinel1|

2. datasets

   A dataset is a specification of a task, along with a set of labels for that task. Currently
   each dataset must correspond to a single collection.

   **Examples**:

   A set of point-detection labels for vessels in Sentinel-1 imagery, or
   a set of attribute labels for vessels within Sentinel-2 imagery, or a set of
   bounding-box labels for vessels in Sentinel-2 imagery.

   **Schema**:

   | cid | name          | type    | notnull | dflt_value | pk  |
   | --- | ------------- | ------- | ------- | ---------- | --- |
   | 0   | id            | INTEGER | 0       |            | 1   |
   | 1   | collection_id | INTEGER | 1       |            | 0   |
   | 2   | name          | TEXT    | 1       |            | 0   |
   | 3   | task          | TEXT    | 1       |            | 0   |
   | 4   | categories    | TEXT    | 0       |            | 0   |

   **Example Row**:
   |id|collection_id|name|task|categories|
   |--|-------|----|----|---|
   |1|1|vessels|point|["vessel"]|

   **Notes**:

   Currently the only supported tasks are `point` (for point detection) and `custom` (for prediction
   of attributes associated with points).

3. images

   An image is a reference to a satellite image, optionally including additional information
   pertaining to how the image is intended to be processed downstream (i.e. in our repo). A single record
   includes information such as the name of an image file (as defined by the provider),
   acquisition time, file format type, location extent, and binned corner coordinates
   corresponding to a specified cooordinate system.

   **Schema**:

   | cid | name         | type     | notnull | dflt_value | pk  |
   | --- | ------------ | -------- | ------- | ---------- | --- |
   | 0   | id           | INTEGER  | 0       |            | 1   |
   | 1   | uuid         | TEXT     | 1       |            | 0   |
   | 2   | name         | TEXT     | 1       |            | 0   |
   | 3   | format       | TEXT     | 1       |            | 0   |
   | 4   | channels     | TEXT     | 1       |            | 0   |
   | 5   | width        | INTEGER  | 1       |            | 0   |
   | 6   | height       | INTEGER  | 1       |            | 0   |
   | 7   | preprocessed | INTEGER  | 1       | 0          | 0   |
   | 8   | hidden       | INTEGER  | 1       | 0          | 0   |
   | 9   | bounds       | TEXT     | 1       |            | 0   |
   | 10  | time         | DATETIME | 1       |            | 0   |
   | 11  | projection   | TEXT     | 0       |            | 0   |
   | 12  | column       | INTEGER  | 1       |            | 0   |
   | 13  | row          | INTEGER  | 1       |            | 0   |
   | 14  | zoom         | INTEGER  | 1       |            | 0   |

   **Example Row**:

   | id  | uuid                             | name                                                                     | format  | channels                                                                                                                                                                                                                                                                                                                  | width | height | preprocessed | hidden | bounds                                                                                                                   | time                      | projection | column | row     | zoom |
   | --- | -------------------------------- | ------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------ | ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------- | ---------- | ------ | ------- | ---- |
   | 1   | a97d69a45df546bb8d02eb6fb794c4ef | S1A_IW_GRDH_1SDV_20211104T170749_20211104T170814_040424_04CAD6_4BAE.SAFE | geotiff | [{"Name":"vh","Count":1},{"Name":"vv","Count":1},{"Name":"vh_overlap0","Count":1},{"Name":"vh_overlap1","Count":1},{"Name":"vh_overlap2","Count":1},{"Name":"vh_overlap3","Count":1},{"Name":"vv_overlap0","Count":1},{"Name":"vv_overlap1","Count":1},{"Name":"vv_overlap2","Count":1},{"Name":"vv_overlap3","Count":1}] | 31428 | 24935  | 0            | 0      | {"Min":{"Lon":-175.24431616814434,"Lat":-18.12758186277979},"Max":{"Lon":-172.54682929070293,"Lat":-16.082187769430707}} | 2021-11-04T17:07:49+00:00 | epsg:3857  | 55407  | 2287033 | 13   |

   **Notes**:

   - There are three distinct but related coordinate systems arising from Web Mercator.

     - One is of course the actual global coordinate system known as Web Mercator and referenced by epsg:3857. The unit of these coordinates is referred to as a meter; more information about maximal bounds and coverage of the coordinate system is documented [here](https://epsg.io/3857). To avoid confusion, we refer to these coordinates as _absolute Web Mercator coordinates_.
     - Another is a coordinate system that can be used to describe discretized (or pixel) coordinates in the space of pixels on a _tiling_ of web mercator. The process goes as follows. Shift absolute Web Mercator coordinates such that they are all positive, and the minimal x and y coordinates are 0. Next, reflect the y coordinates about their midpoint (i.e. y -> y*max - y)). Specify a \_zoom* level between 1 and 24, and a power of 2 called _tile_size_. Partition the portion of the world covered by the Web Mercator projection into (2^(zoom) x 2^(zoom) regular square tiles, with each tile being made up of (tile*size x tile_size pixels), and with the top left corner pixel of one tile coinciding with the (0, 0) coordinate of the translated and reflected absolute web mercator coordinate. We will refer to the pixel coordinates of this tiling as \_pixel coordinates on a Web Mercator tiling (of tileSize 512 and zoom 13)*. These are the coordinates in which most of our metadata is recorded. The function described above to produce these pixel coordinates (and it's inverse) are defined explicitly in the functions `crs_to_pixel` and `pixel_to_crs` of [./data/reports/dataset_report.ipynb](./data/reports/dataset_report.ipynb). See [here](https://learn.microsoft.com/en-us/azure/azure-maps/zoom-levels-and-tile-grid?tabs=csharp) for a more comprehensive introduction to the relevant concepts.
     - Finally, we may sometimes have occasion to refer to the coordinates of entire _tiles_ in the previously described tiling. Whenever we do so, we will refer to these coordinates
       as _tile coordinates on a web mercator tiling (of zoom 13)_

   - Projection indicates a coordinate system that is used to describe the image bounds, and objects relative to those bounds,
     in all references in our source code to the image. The only coordinate system in use here currently is Web Mercator,
     or epsg:3857. See [here](https://en.wikipedia.org/wiki/Web_Mercator_projection) for more information about this coordinate system.
   - Zoom indicates the relative scale of a set of regular 512x512 "tiles" with which we partition the earth. A higher zoom level indicates more tiles are required to partition the earth (precisely, (2^(zoom) tiles) x (2^(zoom) tiles) are used to partition the earth). Currently, the only zoom level used throughout the metadata we provide is 13. It is relevant
     when considering pixel coordinates on a Web Mercator tiling.
   - UUID is a text identifier for an image. In the case of Sentinel-1, the uuids in the metadata correspond to the product uuid assigned by Copernicus API to the image. The uuids here are used
     by our pre-processing logic to construct filesystem directories.
   - Time corresponds to the time at which the satellite acquired the image, in ISO 8601 format.
   - Channels is used internally by the training logic, and includes both the raw channels included
     in the imagery product (i.e. VV, VH), as well as indications of overlap channels produced by our pre-processing logic.
   - Width, height specify the width and height of the image _in pixel coordinates in the relevant Web Mercator tiling_.
   - Column, row: the minimal x and y coordinates of the image extent in _in pixel coordinates in the relevant Web Mercator tiling_.

4. windows

   A window is a cropped subset of an image.

   **Schema**:

   | cid | name       | type    | notnull | dflt_value | pk  |
   | --- | ---------- | ------- | ------- | ---------- | --- |
   | 0   | id         | INTEGER | 0       |            | 1   |
   | 1   | dataset_id | INTEGER | 1       |            | 0   |
   | 2   | image_id   | INTEGER | 1       |            | 0   |
   | 3   | column     | INTEGER | 1       |            | 0   |
   | 4   | row        | INTEGER | 1       |            | 0   |
   | 5   | width      | INTEGER | 1       |            | 0   |
   | 6   | height     | INTEGER | 1       |            | 0   |
   | 7   | hidden     | INTEGER | 1       | 0          | 0   |
   | 8   | split      | TEXT    | 1       | 'default'  | 0   |

   **Example Row**:
   |id|dataset_id|image_id|column|row|width|height|hidden|split|
   |--|----------|--------|------|---|-----|------|------|-----|
   |1|1|51|2718762|1773812|945|945|0|nov-2021-point-train|

   **Notes**:

   - Column and row: the minimal x and y coordinates of a section of an image, _in pixel coordinates in the relevant Web Mercator tiling_.
   - Width and height: widht and height of a section of an image, _in pixel coordinates in the relevant Web Mercator tiling_.
   - Split: An aribtrary text label that can be applied to a window. Used to break
     datasets into pieces based on properties of interest (e.g. acquisition time),
     for e.g. train and validation sets.

5. labels

   A label is metadata specifying some information about a window, such as a point
   that corresponds to the center of an object, or a set of numerical attributes associated
   with a crop.

   **Schema**:

   | cid | name       | type    | notnull | dflt_value | pk  |
   | --- | ---------- | ------- | ------- | ---------- | --- |
   | 0   | id         | INTEGER | 0       |            | 1   |
   | 1   | window_id  | INTEGER | 1       |            | 0   |
   | 2   | column     | INTEGER | 0       |            | 0   |
   | 3   | row        | INTEGER | 0       |            | 0   |
   | 4   | width      | INTEGER | 0       |            | 0   |
   | 5   | height     | INTEGER | 0       |            | 0   |
   | 6   | extent     | TEXT    | 0       |            | 0   |
   | 7   | value      | FLOAT   | 0       |            | 0   |
   | 8   | properties | TEXT    | 0       |            | 0   |

   **Example Row**:
   |id|window_id|column|row|width|height|extent|value|properties|
   |--|---------|------|---|-----|------|------|-----|----------|
   |1|101|3316376|2046567||||||

   **Notes**:

   - Column and row: the x and y coordinates of a point in a window, _in pixel coordinates in the relevant Web Mercator tiling_.
   - Width and height: widht and height of a section of an image, _in pixel coordinates in the relevant Web Mercator tiling_.
   - Properties: A bracket enclosed blob containing numerical properties associated with a point in an image. For example, some of the points at which vessels are labeled in this dataset have corresponding labels for attributes. These are denoted with properties set to values such as `{"Length":85,"Width":17,"Heading":71,"ShipAndCargoType":70,"Speed":0.4000000059604645}`.

</details>
<br>

### Imagery

If you'd like to use the Sentinel-1 dataset (e.g. to re-train our Sentinel-1 models), you'll need access to the full dataset, including both the metadata/labels provided
with this repository, _and_ the raw Sentinel-1 imagery referenced in our metadata.

**Note:** The raw imagery data occupies roughly 3.5TB of space.

#### Acquiring the Imagery

The script `download_imagery.py` allows one to acquire the raw (Sentinel-1) imagery referenced in our annotated dataset. It has
a few command line options, which you can read about in the source file, or using

```bash
python3 download_imagery.py --help
```

#### How to Download Sentinel-1 Training Data

<details>
<summary><b>Manually</b> </summary>

To get started in your own environment:

1. Create and activate a virtual environment with python 3.8.10,

   E.g. run:

   ```bash
   python3.8 -m venv .env
   source .env/bin/activate
   ```

2. Install gdal 3.4.3 on your machine.

   E.g. on Ubuntu 20.04 run:

   ```bash
   sudo apt-get update && \
   sudo apt-get install software-properties-common -y && \
   sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
   sudo apt-get update && sudo apt-get install gdal-bin libgdal-dev libpq-dev python3-gdal python3.8-venv -y
   ```

3. Install the python packages required for downloading the data via pip:

   ```bash
   pip install -r download_requirements.txt
   ```

4. Create a [Copernicus Hub](https://scihub.copernicus.eu/dhus/#/home) account.

5. Export your user credentials and run the download script.

   ```bash
   export COPERNICUS_USERNAME=your-username
   export COPERNICUS_PASSWORD=your-password
   python download_imagery.py \
   --save_dir="./data/training_data" \
   --db_path="./data/metadata.sqlite3"
   ```

   Here the arg to `--save_dir` specifies the relative path to a directory in which all downloaded
   imagery will be stored, and `--db_path` specifies the relative path to the sqlite3 metadata file distributed with this repo.

You should see all Sentinel-1 images referenced in our annotated metadata begin downloading to your specified `save_dir`. The logs will indicate the status of the downloading procedure. When completed, you should have three new directories in your `--save_dir`: _preprocess_, _images_, and _image_cache_. _preprocess_ contains preprocessed (i.e. warped and cropped) copies of the raw Sentinel-1 imagery to which our metadata points. It is all that is strictly necessary for training. _image_cache_ contains the raw Sentinel-1 imagery product archives.

</details>

<details>
<summary><b>Via Docker</b></summary>

You can also use our docker container to avoid setting up the necessary environment manually.

**Note:** Our docker images implicitly assume an x86 architechture. While you may have luck building these images on other architechtures ( e.g. by using docker's `--platform` arg), we have not tested this functionality. For example, we have not tested these builds on ARM machines.
If you have trouble building the image from source on your machine, you can pull it from ghcr (assuming you have copied this repository to GitHub, and have triggered the `build_and_push_docker_images` workflow from the commit of interest), try using the scripts without docker,
or use an x86 VM to build.

To use the docker image:

1.  Build the docker image.

    ```bash
    docker build . -f Dockerfile -t vessel-detection-sentinels
    ```

2.  Create a [Copernicus Hub](https://scihub.copernicus.eu/dhus/#/home) account, and export your user credentials:

    ```bash
    export COPERNICUS_USERNAME=your-username
    export COPERNICUS_PASSWORD=your-password
    ```

3.  Run the docker image.

        ```bash
        docker run -d -v /path/to/host/dir:/home/vessel_detection/data \
        -e COPERNICUS_USERNAME -e COPERNICUS_PASSWORD \
        vessel-detection-download:$IMAGE_TAG \
        --save_dir="/home/vessel_detection/data/training_data" \
        --db_path="/home/vessel_detection/data/metadata.sqlite3"
        ```

        As before, the arg to `--save_dir` specifies the relative path to a directory in which all downloaded
        imagery will be stored, and `--db_path` specifies the relative path to the sqlite3 metadata file distributed with this repo. In the example command shown here, we've pointed these to paths in the container that correspond to a mounted volume.

    </details>
    <br>

## **Training the Sentinel-1 Models**

Once you have all Sentinel-1 imagery associated with our metadata downloaded, you can train our Sentinel-1 detection and attribute prediction models.

**Note:** Training currently requires GPU, due to use of torch's `GradScaler` for gradient scaling. The use of
gradient scaling and automatic mixed precision can be removed from the logic easily by the interested user, but
given the affect on performance, and that the time required to train on CPU is prohibitively long in any case,
we hardcode the use of a GPU device in the training scheme.

The script `train.py` allows one to train our models on our annotated dataset. It has
a few command line options, which you can read about in the source file, or using

```bash
python3 train.py --help
```

### System Requirements and Performance

To train our Sentinel-1 models, we recommend using a system with:

- An x86 architecture.
- A GPU with >= 16GB memory.
- RAM >= 32GB.

These are the specs on which functionality has been tested, though they are not necessarily minimal.

To provide a sense for expected training time, we report total training time for each training
run specified in the default config [./data/cfg/](./data/config) on a particular machine.

| Instance Type                                                                                     | RAM  | GPU Model                                                              | GPU Memory | model type                 | model and dataset config                 | epochs | training time (hours) |
| ------------------------------------------------------------------------------------------------- | ---- | ---------------------------------------------------------------------- | ---------- | -------------------------- | ---------------------------------------- | ------ | --------------------- |
| [GCP n1-highmem-8](https://cloud.google.com/compute/docs/general-purpose-machines#n1-high-memory) | 52GB | [NVIDIA T4](https://cloud.google.com/compute/docs/gpus#nvidia_t4_gpus) | 16GB       | detector (frcnn_cmp2)      | train_sentinel1_detector.json            | 3      | 6                     |
| [GCP n1-highmem-8](https://cloud.google.com/compute/docs/general-purpose-machines#n1-high-memory) | 52GB | [NVIDIA T4](https://cloud.google.com/compute/docs/gpus#nvidia_t4_gpus) | 16GB       | attribute predictor (attr) | train_sentinel1_attribute_predictor.json | 80     | 4                     |

We describe below how to set up an environment on your machine for training (either manually, or via docker).

**Note:** Our docker images implicitly assume an x86 architecture. While you may have luck building these images on other architechtures ( e.g. by using docker's `--platform` arg),
we have not tested this functionality. For example, we have not tested these builds on ARM machines.
If you have trouble building the image from source on your machine, you can pull it from ghcr, use the scripts without docker,
or use an x86 VM to build.

### How to Train

<details>
<summary><b>Manually</b></summary>

To get started in your own environment:

1. Create and activate a virtual environment with python 3.8.10.

   e.g. via

   ```bash
   python3.8 -m venv .env
   source .env/bin/activate
   ```

2. Install gdal 3.4.3 on your system.

   E.g. on Ubuntu 20.04 run:

   ```bash
   sudo apt-get update && \
   sudo apt-get install software-properties-common -y && \
   sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
   sudo apt-get update && sudo apt-get install gdal-bin libgdal-dev libpq-dev python3-gdal python3.8-venv -y
   ```

3. Install the python packages required for training via pip:

   ```bash
   pip install -r train_requirements.txt
   ```

4. Place the entire `preprocess` folder generated by the data download script, as well as the [`./data/metadata.sqlite3`](./data/metadata.sqlite3) file, somewhere accessible on your machine.

   In what follows we'll assume we've moved both of these things to the `./data` folder in this repository.

5. Run the training script for the detection model.

   Logging training metrics to a [WANDB](https://wandb.ai/site) account:

   ```bash
       export RUN_TAG=$(git rev-parse HEAD)
       export WANDB_MODE=online
       export WANDB_API_KEY=your-api-key
       python train.py \
           --config_path ./data/cfg/train_sentinel1_detector.json \
           --training_data_dir ./data \
           --save_dir ./data/models \
           --metadata_path ./data/metadata.sqlite3
   ```

   Logging training metrics only locally:

   ```bash
      export RUN_TAG=$(git rev-parse HEAD)
      python train.py \
          --config_path ./data/cfg/train_sentinel1_detector.json \
          --training_data_dir ./data \
          --save_dir ./data/models \
          --metadata_path ./data/metadata.sqlite3
   ```

6. Run the training script for the attribute prediction model.

   Logging training metrics to a [WANDB](https://wandb.ai/site) account:

   ```bash
       export RUN_TAG=$(git rev-parse HEAD)
       export WANDB_MODE=online
       export WANDB_API_KEY=your-api-key
       python train.py \
           --config_path ./data/cfg/train_sentinel1_attribute_predictor.json \
           --training_data_dir ./data \
           --save_dir ./data/models \
           --metadata_path ./data/metadata.sqlite3
   ```

   Logging training metrics only locally:

   ```bash
       export RUN_TAG=$(git rev-parse HEAD)
       python train.py \
           --config_path ./data/cfg/train_sentinel1_attribute_predictor.json \
           --training_data_dir ./data \
           --save_dir ./data/models \
           --metadata_path ./data/metadata.sqlite3
   ```

As training progresses for each model, the logs will indicate the status of the run.

When training for a model completes, you should see the `--save_dir` you specified populated with a directory named `{model_name}/{RUN_TAG}`. This directory will house
saved weights, and training and model config. Here `{model_name}` will indicate whether the model being trained is an instance of the detection model (the default config uses
`model_name=frcnn_cmp2` for that), or an instance of the attribute prediction model (the default config uses `model_name=attr` for that).

</details>

<details>
<summary><b>Via Docker</b></summary>

You can also use our docker container to avoid setting up the necessary environment manually.

1.  Build the docker image.

    ```bash
    export IMAGE_TAG=$(git rev-parse --short HEAD)
    docker build . -f Dockerfile -t vessel-detection-sentinels:$IMAGE_TAG
    ```

2.  Place the entire `preprocess` folder generated by the data download script, as well as the [`./data/metadata.sqlite3`](./data/metadata.sqlite3) file, somewhere accessible on your machine.

    In what follows we'll assume we've moved both things to the `/path/to/your/data` folder on our machine.

3.  Run the training container for the detector:

    Logging training metrics via a WANDB account:

    ```bash
    export RUN_TAG=$(git rev-parse HEAD)
    export WANDB_MODE=online
    export WANDB_API_KEY=your-api-key
    docker run --shm-size 16G --gpus='"device=0"' -d \
    -v /path/to/your/data:/home/vessel_detection/data \
    -e RUN_TAG=$RUN_TAG \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_MODE=$WANDB_MODE \
    vessel-detection-train:$IMAGE_TAG \
    --config_path /home/vessel_detection/data/cfg/train_sentinel1_detector.json \
    --training_data_dir /home/vessel_detection/data \
    --save_dir /home/vessel_detection/data/models \
    --metadata_path /home/vessel_detection/data/metadata.sqlite3
    ```

    or without WANDB:

    ```bash
    export RUN_TAG=$(git rev-parse HEAD)
    export WANDB_MODE=offline
    docker run --shm-size 16G --gpus='"device=0"' -d \
    -v /path/to/your/data:/home/vessel_detection/data \
    -e RUN_TAG=$RUN_TAG \
    -e WANDB_MODE=$WANDB_MODE \
    vessel-detection-train:$IMAGE_TAG \
    --config_path /home/vessel_detection/data/cfg/train_sentinel1_detector.json \
    --training_data_dir /home/vessel_detection/data \
    --save_dir /home/vessel_detection/data/models \
    --metadata_path /home/vessel_detection/data/metadata.sqlite3
    ```

4.  Run the training container for the attribute predictor:

        Logging training metrics via a WANDB account:
        ```bash
        export RUN_TAG=$(git rev-parse HEAD)
        export WANDB_MODE=online
        export WANDB_API_KEY=your-api-key
        docker run --shm-size 16G --gpus='"device=0"' -d \
        -v /path/to/your/data:/home/vessel_detection/data \
        -e RUN_TAG=$RUN_TAG \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -e WANDB_MODE=$WANDB_MODE \
        vessel-detection-train:$IMAGE_TAG \
        --config_path /home/vessel_detection/data/cfg/train_sentinel1_attribute_predictor.json \
        --training_data_dir /home/vessel_detection/data \
        --save_dir /home/vessel_detection/data/models \
        --metadata_path /home/vessel_detection/data/metadata.sqlite3
        ```

        or without WANDB:

        ```bash
        export RUN_TAG=$(git rev-parse HEAD)
        export WANDB_MODE=offline
        docker run --shm-size 16G --gpus='"device=0"' -d \
        -v /path/to/your/data:/home/vessel_detection/data \
        -e RUN_TAG=$RUN_TAG \
        -e WANDB_MODE=$WANDB_MODE \
        vessel-detection-train:$IMAGE_TAG
        --config_path /home/vessel_detection/data/cfg/train_sentinel1_attribute_predictor.json \
        --training_data_dir /home/vessel_detection/data \
        --save_dir /home/vessel_detection/data/models \
        --metadata_path /home/vessel_detection/data/metadata.sqlite3
        ```

    As training progresses for each model, the logs will indicate the status of the run.

When training for a model completes, you should see the `--save_dir` you specified populated with a directory named `{model_name}/{RUN_TAG}`. This directory will house
saved weights, and training and model config. Here `{model_name}` will indicate whether the model being trained is an instance of the detection model (the default config uses
`model_name=frcnn_cmp2` for that), or an instance of the attribute prediction model (the default config uses `model_name=attr` for that).

</details>

<br>

## **Inference**

You can perform inference right off the bat with our provided trained weights, or with weights you produce from running our training scripts.

The script `src/main.py` allows one to perform inference on a raw (but de-compressed) Sentinel-1 and Sentinel-2 image products.

### System Requirements and Performance

To perform inference with trained copies of our models, we recommend using a system with:

- An x86 architecture.
- A GPU with >= 8GB memory.
- RAM >= 32GB.

The lowest values reported here are the specs on which functionality has been tested, but are not necessarily the minimal requirements.

To provide a sense for expected inference time, we report total inference time on two particular particular machines,
on representative Sentinel-1 scenes. Run time and resource requirements for our Sentinel-2 models are similar.

| Instance Type                                                                                     | RAM  | GPU Model                                                              | GPU Memory | inference time (seconds) | using cpu only |
| ------------------------------------------------------------------------------------------------- | ---- | ---------------------------------------------------------------------- | ---------- | ------------------------ | -------------- |
| [GCP n1-highmem-8](https://cloud.google.com/compute/docs/general-purpose-machines#n1-high-memory) | 52GB | [NVIDIA T4](https://cloud.google.com/compute/docs/gpus#nvidia_t4_gpus) | 16GB       | 307                      | false          |
| AMD Ryzen 9 3900X 12-Core Processor                                                               | 64GB | NVIDIA RTX 2070 SUPER                                                  | 8GB        | 174                      | false          |
| AMD Ryzen 9 3900X 12-Core Processor                                                               | 64GB | NVIDIA RTX 2070 SUPER                                                  | 8GB        | 2100                     | true           |

We describe below how to set up an environment on your machine for inference (either manually, or via docker).

### How to Make Predictions

<details>
<summary> <b>Manually</b> </summary>

### Sentinel-1

To get started in your own environment:

1.  Download a Sentinel-1 scene (and optionally two historical overlaps of that scene) from the Copernicus Hub UI or API, and place it in a folder of your choosing.

    For example, let's suppose we've downloaded three such scenes which we've placed on our system at:

    - `./data/S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE`
    - `./data/S1B_IW_GRDH_1SDV_20211118T025212_20211118T025237_029636_03896D_FCBE.SAFE`
    - `./data/S1B_IW_GRDH_1SDV_20211106T025212_20211106T025237_029461_03840B_6E73.SAFE`

2.  Collect a directory of trained Senntinel-1 model artifacts on your machine. At minimum,
    you will need one set of trained weights and model config for the detection model and one set for the attribute prediction model.
    You can obtain these directories by training the models yourselves, or by using pre-trained weights found in data/model_artifacts.

        For example, let's suppose we're using the pre-trained Sentinel-1 artifacts distributed separately, and
        we've copied the following directory structure with these artifacts into the `./data` folder of this repo:

        ```
        ./data
            |----attr
            |      |---c34aa37
            |              |-----best.pth
            |              |-----cfg.json
            |
            |---frcnn_cmp2
                    |
                    |--3dff445
                          |-----best.pth
                          |-----cfg.json
        ```

3.  Create and activate a virtual environment with python 3.8.10.

    e.g. via

    ```bash
    python3.8 -m venv .env
    source .env/bin/activate
    ```

4.  Install gdal 3.4.3 on your system.

    E.g. on Ubuntu 20.04 run:

    ```bash
    sudo apt-get update && \
    sudo apt-get install software-properties-common -y && \
    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    sudo apt-get update && apt-get install gdal-bin libgdal-dev libpq-dev python3-gdal python3.8-venv -y
    ```

5.  Install the python packages required for training via pip:

    ```bash
    pip install -r requirements.txt
    ```

6.  Run the inference script for the detection model.

    To run inference on a single image, without access to historical overlaps:

### Sentinel-2

1.  Download a Sentinel-2 scene (and optionally two historical overlaps of that scene) from the Copernicus Hub UI or API, and place it in a folder of your choosing.

        For example, let's suppose we've downloaded three such scenes which we've placed on our system at:

        -  `./data/S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE`
        -  `./data/S2A_MSIL1C_20230111T061221_N0509_R134_T42RUN_20230111T064344.SAFE`
        -  `./data/S2B_MSIL1C_20230106T061239_N0509_R134_T42RUN_20230106T063926.SAFE`

2.  Collect a directory of trained Sentinel-2 model artifacts on your machine. At minimum,
    you will need one set of trained weights and model config for the detection model and one set for the attribute prediction model.
    You can obtain these directories by training the models yourselves, or by using pre-trained weights we will provide separately.

        For example, let's suppose we're using the pre-trained Sentinel-2 artifacts distributed separately, and
        we've copied the following directory structure with these artifacts into the `./data` folder of this repo:

        ```
        ./data
            |----attr
            |      |---e609150-sentinel2-attr-resnet
            |              |-----best.pth
            |              |-----cfg.json
            |
            |---frcnn_cmp2
                    |
                    |--15cddd5-sentinel2-swinv2-small-fpn
                        |-----best.pth
                        |-----cfg.json
        ```

3.  Create and activate a virtual environment with python 3.8.10.

    e.g. via

    ```bash
    python3.8 -m venv .env
    source .env/bin/activate
    ```

4.  Install gdal 3.4.3 on your system.

    E.g. on Ubuntu 20.04 run:

    ```bash
    sudo apt-get update && \
    sudo apt-get install software-properties-common -y && \
    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    sudo apt-get update && apt-get install gdal-bin libgdal-dev libpq-dev python3-gdal python3.8-venv -y
    ```

5.  Install the python packages required for training via pip:

    ```bash
    pip install -r requirements.txt
    ```

6.  Run the inference script for the detection model.

    To run inference on a single image, without access to historical overlaps:

    ### Notes (common to both imagery sources):

    Here `--raw_path` must point to a directory containing the directory specified by `--scene_id` (and `--historical1` and `--historical2`, if they are provided).

    The model weight dirs for the detector and attribute predictor models are respectively pointed to by `--detector_model_dir` and `--postprocess_model_dir`.

    The outputs of the detection script will be populated in the directory specified by `--output`.

    If you'd prefer to run the inference script on CPU, you can optionally pass the flag `--force-cpu`. However, please note that running inference
    on CPU is currently prohibitively slow. For example, while inference with a GPU might take ~2 mins for a Sentinel-1 scene, it might take
    closer to ~30 mins to perform the same call on CPU.

</details>

<details>
<summary> <b>Via Docker</b> </summary>

You can also use our docker container to avoid setting up the necessary environment manually.

**Note:** Our docker images implicitly assume an x86 architecture. While you may have luck building these images on other architectures ( e.g. by using docker's `--platform` arg), we have not tested this functionality. For example, we have not tested these builds on ARM machines.
If you have trouble building the image from source on your machine, you can pull it from ghcr, use the scripts without docker,
or use an x86 VM to build.

### Sentinel-1

1.  Acquire the docker image.

    Pull the latest container from Github container registry:

    ```bash
    export REPO_OWNER=your-repo-owner-name
    export REPO_NAME=sentinel-vessel-detection
    export IMAGE_TAG=latest
    docker pull ghcr.io/$REPO_OWNER/$REPO_NAME/vessel-detection:$IMAGE_TAG
    ```

    or build it from source by running the following from the top level of this repo:

    ```bash
    export IMAGE_TAG=$(git rev-parse --short HEAD)
    docker build . -f Dockerfile -t vessel-detection:$IMAGE_TAG
    ```

2.  Prepare a machine with at least 16GB RAM, and a GPU w/ >= 8GB memory.

3.  Download a Sentinel-1 scene (and optionally up to two historical overlaps) from the Copernicus Hub UI or API, and place it in a folder of your choosing.

    For example, let's suppose we've downloaded three such scenes which we've placed on our system at:

    - `/path/to/your/data/S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE`
    - `/path/to/your/data/S1B_IW_GRDH_1SDV_20211118T025212_20211118T025237_029636_03896D_FCBE.SAFE`
    - `/path/to/your/data/S1B_IW_GRDH_1SDV_20211106T025212_20211106T025237_029461_03840B_6E73.SAFE`

4.  Collect a directory of trained Sentinel-1 model artifacts on your machine. At minimum,
    you will need one set of trained weights and model config for the detection model and one set for the attribute prediction model.
    You can obtain these directories by training the models yourselves, or by using pre-trained weights we will provide separately.

        For example, let's suppose we're using the pre-trained artifacts distributed separately, and
        we've copied the following directory structure with these artifacts into the `/path/to/your/data` folder of this repo:

        ```
        /path/to/your/data
            |----attr
            |      |---c34aa37
            |              |-----best.pth
            |              |-----cfg.json
            |
            |---frcnn_cmp2
                    |
                    |--3dff445
                          |-----best.pth
                          |-----cfg.json
        ```

5.  On that machine,

    To run inference on a single image, without access to historical overlaps, run:

    ```bash
    docker run --shm-size 16G --gpus='"device=0"' \
    -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
    --raw_path=/home/vessel_detection/data/ \
    --scratch_path=/home/vessel_detection/data/scratch/ \
    --output=/home/vessel_detection/data/output/ \
    --detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/3dff445 \
    --postprocess_model_dir=/home/vessel_detection/data/models/attr/c34aa37 \
    --scene_id=S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE \
    --conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel1
    ```

    To run inference on a single image, using additional context from a single historical overlap of the image:

    ```bash
    docker run --shm-size 16G --gpus='"device=0"' \
    -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
    --raw_path=/home/vessel_detection/data/ \
    --scratch_path=/home/vessel_detection/data/scratch/ \
    --output=/home/vessel_detection/data/output/ \
    --detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/3dff445 \
    --postprocess_model_dir=/home/vessel_detection/data/models/attr/c34aa37 \
    --historical1=S1B_IW_GRDH_1SDV_20211118T025212_20211118T025237_029636_03896D_FCBE.SAFE \
    --scene_id=S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE \
    --conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel1
    ```

    To run inference on a single image, using additional context from two historical overlaps of the image:

    ```bash
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
    ```

### Sentinel-2

1.  Acquire the docker image.

    Pull the latest container from Github container registry:

    ```bash
    export REPO_OWNER=your-repo-owner-name
    export REPO_NAME=sentinel-vessel-detection
    export IMAGE_TAG=latest
    docker pull ghcr.io/$REPO_OWNER/$REPO_NAME/vessel-detection:$IMAGE_TAG
    ```

    or build it from source by running the following from the top level of this repo:

    ```bash
    export IMAGE_TAG=$(git rev-parse --short HEAD)
    docker build . -f docker/inference.dockerfile -t vessel-detection:$IMAGE_TAG
    ```

2.  Prepare a machine with at least 16GB RAM, and a GPU w/ >= 8GB memory.

3.  Download a Sentinel-2 scene (and optionally up to two historical overlaps) from the Copernicus Hub UI or API, and place it in a folder of your choosing.

    For example, let's suppose we've downloaded three such scenes which we've placed on our system at:

    - `/path/to/your/data/S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE`
    - `/path/to/your/data/S2A_MSIL1C_20230111T061221_N0509_R134_T42RUN_20230111T064344.SAFE`
    - `/path/to/your/data/S2B_MSIL1C_20230106T061239_N0509_R134_T42RUN_20230106T063926.SAFE`

4.  Collect a directory of trained Sentinel-2 model artifacts on your machine. At minimum,
    you will need one set of trained weights and model config for the detection model and one set for the attribute prediction model.
    You can obtain these directories by training the models yourselves, or by using pre-trained weights we will provide separately.

        For example, let's suppose we're using the pre-trained artifacts distributed separately, and
        we've copied the following directory structure with these artifacts to `path/to/your/data` folder on your machine:

        ```
        path/to/your/data
            |----attr
            |      |---e609150-sentinel2-attr-resnet
            |              |-----best.pth
            |              |-----cfg.json
            |
            |---frcnn_cmp2
                    |
                    |--15cddd5-sentinel2-swinv2-small-fpn
                        |-----best.pth
                        |-----cfg.json
        ```

5.  On that machine,

        To run inference on a single image, without access to historical overlaps, run:
        ```bash
        docker run --shm-size 16G --gpus='"device=0"' \
        -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
        --raw_path=/home/vessel_detection/data/ \
        --scratch_path=/home/vessel_detection/data/scratch/ \
        --output=/home/vessel_detection/data/output/ \
        --detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/15cddd5-sentinel2-swinv2-small-fpn \
        --postprocess_model_dir=/home/vessel_detection/data/models/attr/e609150-sentinel2-attr-resnet \
        --scene_id=S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE \
        --conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel2
        ```

        To run inference on a single image, using additional context from a single historical overlap of the image:
        ```bash
        docker run --shm-size 16G --gpus='"device=0"' \
        -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
        --raw_path=/home/vessel_detection/data/ \
        --scratch_path=/home/vessel_detection/data/scratch/ \
        --output=/home/vessel_detection/data/output/ \
        --detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/15cddd5-sentinel2-swinv2-small-fpn \
        --postprocess_model_dir=/home/vessel_detection/data/models/attr/e609150-sentinel2-attr-resnet \
        --historical1=S2A_MSIL1C_20230111T061221_N0509_R134_T42RUN_20230111T064344.SAFE \
        --scene_id=S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE \
        --conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel2
        ```

        To run inference on a single image, using additional context from two historical overlaps of the image:
        ```bash
        docker run --shm-size 16G --gpus='"device=0"' \
        -v /path/to/your/data:/home/vessel_detection/data vessel-detection:$IMAGE_TAG \
        --raw_path=/home/vessel_detection/data/ \
        --scratch_path=/home/vessel_detection/data/scratch/ \
        --output=/home/vessel_detection/data/output/ \
        --detector_model_dir=/home/vessel_detection/data/models/frcnn_cmp2/15cddd5-sentinel2-swinv2-small-fpn \
        --postprocess_model_dir=/home/vessel_detection/data/models/attr/e609150-sentinel2-attr-resnet \
        --historical1=S2A_MSIL1C_20230111T061221_N0509_R134_T42RUN_20230111T064344.SAFE \
        --historical2=S2B_MSIL1C_20230106T061239_N0509_R134_T42RUN_20230106T063926.SAFE \
        --scene_id=S2A_MSIL1C_20230108T060231_N0509_R091_T42RUN_20230108T062956.SAFE \
        --conf=.9 --nms_thresh=10 --save_crops=True --catalog=sentinel2
        ```

        ### Notes (common to both imagery sources):

        Here `--raw_path` must point to a directory containing the file specified by `--scene_id` (and ``--historical1`` and ``--historical2``, if they are provided).

        The model weight dirs for the detector and attribute predictor models are respectively pointed to by `--detector_model_dir` and `--postprocess_model_dir`.

        The outputs of the detection script will be populated in the directory specified by `--output`.

        If you'd prefer to run the inference script on CPU, you can optionally pass the flag `--force-cpu`. However, please note that running inference
        on CPU is currently prohibitively slow. For example, while inference with a GPU might take ~2 mins for a Sentinel-1 scene, it might take
        closer to ~30 mins to perform the same call on CPU.

    </details>
    <br>

### Inference Outputs

Running the inference script will produce at most three types of artifact in the specified output directories:

1.  A csv, `predictions.csv`, with one row for each detected vessel.

    Here is the schema of the output csv:

    | **column name**   | **description**                                                                                                                                                                                                                                                                                                                            |
    | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | preprocess_row    | The x coordinate of the detection, in pixel coordinates in the _preprocessed_ imagery (saved to the output scratch dir).                                                                                                                                                                                                                   |
    | preprocess_column | The y coordinate of the detection, in pixel coordinates in the _preprocessed_ imagery (saved to the output scratch dir).                                                                                                                                                                                                                   |
    | lat               | Latitude associated with the detection.                                                                                                                                                                                                                                                                                                    |
    | lon               | Longitude associated with the detection.                                                                                                                                                                                                                                                                                                   |
    | score             | Float between 0 and 1 representing the confidence that a vessel was detected at the specified location in the image.                                                                                                                                                                                                                       |
    | vessel_length_m   | Vessel length (meters).                                                                                                                                                                                                                                                                                                                    |
    | vessel_width_m    | Vessel width (meters).                                                                                                                                                                                                                                                                                                                     |
    | heading_bucket_i  | Probability that the heading direction of the detected vessel lies in the range between i*(360/16) and (i + 1)* (360/16) degrees, measured clockwise relative to true north.                                                                                                                                                               |
    | vessel_speed_k    | Vessel speed (knots).                                                                                                                                                                                                                                                                                                                      |
    | is_fishing_vessel | Float between 0 and 1 representing the probability that the detected vessel is a fishing vessel.                                                                                                                                                                                                                                           |
    | detect_id         | Same as scene*id below, but with a zero-indexed `*{idx}` string appended to the name, specifying the count of detections at the time this row was written.                                                                                                                                                                                 |
    | scene_id          | The provider defined name of the scene on which detection was performed. See e.g. https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-product-formatting for Sentinel-1 and https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention for Sentinel-2. |
    | row               | The x coordinate of the detection, in pixel coordinates in the original imagery.                                                                                                                                                                                                                                                           |
    | column            | The y coordinate of the detection, in pixel coordinates in the original imagery .                                                                                                                                                                                                                                                          |
    | orientation       | The clockwise rotation angle (in degrees) necessary to rotate an upwards pointing vertical line in the image to align with North                                                                                                                                                                                                           |
    | meters_per_pixel  | The spatial resolution (in meters) of a single pixel in the output crops associated with a detection.                                                                                                                                                                                                                                      |

2.  Cropped PNG images (of size 128x128) surrounding each vessel detection, for certain subsets of spectral channels in the input imagery. These are only produced if inference is run with `--save_crops` set to True. If produced,
    these files are named according to the convention: `{scene_id}_{detection_idx}_{channel_subset_abbreviation}.png`.

        - For Sentinel-1, we output two crops, one for each of the imagery channels. Here is a sample of the VH and VV polarization channels for the cropped outputs of a vessel detected in Sentinel-1 imagery:

        <figure><img src=./data/example_output/sentinel1/S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE_0_vh.png alt="Detection Crop in VH Channel" style="width:48%">
        <img src=./data/example_output/sentinel1/S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE_0_vv.png alt="Detection Crop in VV Channel" style="width:48%"><br></figure>
        <figcaption align = "center"><b>A Sentinel-1 vessel detection</b>. <br> <b>Left: VH polarization. Right: VV polarization.</b></figcaption>
        <br>

        - For Sentinel-2, we output at most 4 crops, one for the TCI channel collection, and one each for bands 8, 11, 12, if they are used by the model in question. Here is a sample of the TCI image corresponding to the cropped output of a vessel detected in Sentinel-2 imagery:

        <figure align = "center" ><img src=./data/example_output/sentinel2/S2B_MSIL1C_20211102T061019_N0301_R134_T42RUN_20211102T071305.SAFE_0_tci.png alt="Detection Crop in TCI channel" style="width:60%">
        <figcaption align = "center"><b>A Sentinel-2 vessel detection</b>. <br> <b> TCI Channels.</b></figcaption>
        <br>
        </figure>

3.  Warped copies of the relevant TIFF files in the input, and numpy arrays containing processed variants of the input images,
    in the scratch folder specified.

        By default, these intermediate files get produced in the specified scratch directory at runtime, and then cleaned up
        at the end of the inference call. If you'd like to keep these around to inspect them or use them in some way,
        pass the flag `--keep_intermediate_files` to the detect call.

<br>

## Acknowledgements

1. Skylight-ML (especially Mike Gartner who wrote most of this codebase
2. PRIOR at AI2 (especially Favyen Bastani and Piper Wolters) contributed considerable expertise to the foundational architectures of both models
3. European Space Agency for making Sentinel-1 and Sentinel-2 data available to the public.
4. The [Defense Innovation Unit](https://www.diu.mil/) (DIU) who built the foundation for this work through [xView3](https://iuu.xview.us/).
