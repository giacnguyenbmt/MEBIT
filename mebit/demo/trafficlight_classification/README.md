# Traffic light classification evaluation

## Installation

This demo tests `onnx` models. We recommend that users follow the bellow command to install requirements.

```bash
pip install -r mebit/demo/trafficlight_classification/requirements.txt
```

Download datasets to test the traffic light classfication model.

```bash
gdown 1treeleluT9bcC4VIaYIbBmsDBIMNPdpA
unzip -q -d ./data rotation_compactness_lisa_data.zip
gdown 1FgIOZYCTPMnrndPJQVjtI_XtScH7X1rT
uznip -q -d ./data test_tflight_overal.zip
```

## Usage

Command to run the evaluation process with bellow option:

- "blurring"
- "increasing_brightness"
- "increasing_contrast"
- "decreasing_brightness"
- "decreasing_contrast"
- "down_scale"
- "crop"

```bash
rm -rf result_folder && python -m \
mebit.demo.trafficlight_classification.trafficlight_evaluation \
data/test_tflight_overal/img/ \
data/test_tflight_overal/gt/ \
"" "result_folder"
```

Command to run the evaluation process with bellow option:

- "blurring"
- "increasing_brightness"
- "increasing_contrast"
- "decreasing_brightness"
- "decreasing_contrast"
- "down_scale"
- "crop"

```bash
rm -rf rc_result_folder && python -m \
mebit.demo.trafficlight_classification.trafficlight_evaluation \
data/rotation_compactness_lisa_data/image/ \
data/rotation_compactness_lisa_data/gt/ \
"" "rc_result_folder"
```

`Results are stored in "result_folder" and "rc_result_folder"`
