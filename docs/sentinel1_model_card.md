
# Sentinel-1, Model Variants: 3dff445/c34aa37

This model can be used to detect vessels in synthetic aperture radar (SAR) imagery produced by the Sentinel-1 satellite constellation.

## Model Details

This model is designed to perform inference only on Sentinel-1 products acquired via the [Interferometric Wide Swath
acquisition mode](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath), in the [Level-1 GRD product type](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels), and containing VV and VH polarization channels.

### Model Description

Passing a Sentinel-1 scene to the model produces predictions for the centers of detected vessels, as well as predictions for a set of numerical and categorical attributes associated with each vessel, including length, width, speed, and heading.
- **Developed by:** Skylight ML
- **License:** TBD
- **Model Type** Point Detection

## Uses

### Direct Use

See [./README.md](./README.md) for detailed instructions on usage.

Given an appropriate environment, you can run a provided script to perform inference on a raw scene on your local:

```zsh
python detect.py \
    --raw_path=data/ \
    --scratch_path=data/scratch/ \
    --output=data/output/ \
    --scene_id=S1B_IW_GRDH_1SDV_20211130T025211_20211130T025236_029811_038EEF_D350.SAFE \
    --conf=.9 \
    --nms_thresh=10 \
    --save_crops=True \
    --detector_model_dir=data/models/frcnn_cmp2/3dff445 \
    --postprocess_model_dir=data/models/attr/c34aa37 \
    --catalog=sentinel1
```


## Bias, Risks, and Limitations

In some cases, it is difficult to discern whether a sub-region of a Sentinel-1
image corresponds to a vessel, so model accuracy may be subject to interpretation. In other cases, it may be
_impossible_ to discern whether a sub-region corresponds to a vessel (e.g. if the sub-region occupies one pixel),
and therefore to tell whether the model is acting appropriately. The model learns from visual features,
and so is of course more likely to successfully detect vessels which are more visibly apparent (e.g. those that are larger).
As such, one should not assume that the distribution of detected vessels at all resembles the distributon of _all_
vessels that are actually in the imagery.




## Training Details

There are two separate models being documented here. One is responsible for detecting vessels,
and the other is responsible for predicting numerical and categorical attributes associated with vessels.

The two models are trained separately, with independent configs.

### Training Data

Training data for these models consist of:

-  ~60k annotated vessel locations from Sentinel-1 scenes.
-  ~20k annotated vessel attributes associated with 20k of those vessel locations in Sentinel-1 scenes.

Extensive details about the training data are documented in [./README.md](./README.md).

### Training Procedure 

The training data above is split into train and validation splits. Details about those splits are documented more carefully in [./README.md](./README.md). We typically train via Adam, though the optimizer is configurable.


#### Training Hyperparameters

Training hyperparameters used to produce the 3dff445/1d8989a artifacts are fully specified via configs: [./data/cfg/train_sentinel1_detector.json](./data/cfg/train_sentinel1_detector.json) and [./data/cfg/train_sentinel1_attribute_predictor.json](./data/cfg/train_sentinel1_attribute_predictor.json) respectively.

By default training uses a single GPU and mixed precision fp32/fp16 (via torch automatic mixed precision).

## Evaluation

These models were evaluated on the validation sets specified in their training configs, namely [./data/cfg/train_sentinel1_detector.json](./data/cfg/train_sentinel1_detector.json) and [./data/cfg/train_sentinel1_attribute_predictor.json](./data/cfg/train_sentinel1_attribute_predictor.json) respectively.


### Evaluation Data

For the detection model, the validation data consisted of ~6k annotated vessel locations in Sentinel-1 scenes.

For the attribute prediction model, the validation data consisted of ~1.7k vessel locations in Sentinel-1 scenes additionally annotated with attributes.

#### Metrics

For the detection task, we computed a standard collection of confusion matrix based metrics, across a
range of confidence thresholds, such as:

-  Precision
-  Recall
-  F1


For the attribute prediction task, we computed a variety of metrics, including:

-  Length
    - MAE
    - Average Percent Error Score
-  Width
    - MAE
    - Average Percent Error Score
-  Speed
    - MAE
    - MAE (on ground truth moving vessels)
    - Average Percent Error Score
-  Heading
    - Accuracy

Here, we use the following definitions:

- MAE: mean absolute error
- Average Percent Error Score: Fix a maximum allowable value, `MAX_VAL`. For each prediction/ground truth pair `(x, y)`,
calculate:

    ```
    (x_capped, y_capped) = (min(x, MAX_VAL), min(y, MAX_VAL))
    pct_error = |x_capped - y_capped| / y_capped
    avg_percent_error = AVG(pct_error)
    average_percent_error_score = (1 - min(avg_percent_error/size(validation_set), 1))
    ```

We use `MAX_VAL=500` for length, `MAX_VAL=130` for width, and `MAX_VAL=25` for speed.

### Results

#### Detection
Model artifact: 3dff445 

-  Precision: .836
-  Recall: .819
-  F1: .827


#### Attribute Prediction
Model artifact: c34aa37


-  Length
    - MAE: 27m
    - Average Percent Error Score: .70
-  Width
    - MAE: 4.9m
    - Average Percent Error Score: .76
-  Speed
    - MAE: 1.7 knots
    - MAE (on ground truth moving vessels): 2.37 knots
    - Average Percent Error Score: 0
-  Heading
    - Accuracy: 27.8%
    - Average Degree Error Score: .57
    - Average Degree Error Score, axis: .74


## Environmental Impact (Training)

- **Hardware Type: NVIDIA T4** 
- **Hours used: 202.1**
- **Cloud Provider: GCP** 
- **Compute Region: us-west1-b** 
- **Carbon Emitted: 4.25 kg CO2**\*\*

**Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

### Model Architecture and Objective

The detection model consists of a standard FasterRCNN implementation, with a customized backbone.
The backbone consists of a small, fully-convolutional encoder, whose outputs get passed through a feature
pyramid network. The main customization is that the encoder in the backbone is designed to accept
either:
-  A single image crop of a region.
-  An image crop of a region, along with one crop of the same region from a different time.
-  An image crop of a region, along with two crops of the same region from two different times.

The encoder concatenates the feature array for an image crop of interest with the (aggregate) historical feature arrays if they are present,
and otherwise uses imputed constant arrays as substitutes for historical information. The intent is to allow
the encoder to learn that objects which consistently occur in the same location in historical overlaps are
unlikely to represent transient objects, such as vessels.

The model we use for attribute prediction is a standard implementation of a 50 layer Resnet backbone (see the original resnet [paper](https://arxiv.org/abs/1512.03385)),
with simple heads (consisting of convolutional and fully connected linear layers) constructed to compute each attribute of interest.



## Model Card Authors
Mike Gartner

## Model Card Contact
mikeg@allenai.org
