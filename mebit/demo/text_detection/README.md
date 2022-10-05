# Text detection evalution

## Installation

This demo tests MMOCR models. We recommend that users follow our best practices to install MMOCR.

**Step 0.** Install MMCV using MIM.

```bash
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDetection as a dependency.

```bash
pip install mmdet
```

**Step 2.** Install MMOCR.

```bash
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -r requirements.txt
pip install -v -e .
```

**Step 3.** Clone MEBIT into MMOCR folder.

```bash
git clone https://github.com/giacnguyenbmt/MEBIT.git
pip install -r MEBIT/requirements.txt
```

## Usage

Command to run the evaluation process:

```bash
rm -rf result_folder && python -m MEBIT.mebit.demo.text_detection.tdet_run
```
