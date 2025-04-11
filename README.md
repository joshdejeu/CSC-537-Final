# CSC 537 Final Project
### Deep learning-based image segmentation for garbage detection

## Quick Start (Recommended)
Easiest setup, run everything in one go

1. Install required packages
```bash
pip install -r .\start_here\requirements.txt
```
2. Run all processes in one go
```bash
python .\start_here\run.py
```


## Manual Set Up
If you want to manually run each step for more control

1. Install required packages
```bash
pip install -r .\start_here\requirements.txt
```
2. Optional: Check if PyTorch can access the GPU (CUDA, faster than CPU)
```bash
python .\start_here\utils\check_gpu.py
```

3. Download the dataset
```bash
python .\start_here\setup\download_data.py
```
4. Organize images into their respective folders
```bash
python .\start_here\setup\organize_dataset.py
```
5. Convert COCO-style annotations to YOLOv8 format
```bash
python .\start_here\setup\convert_coco_to_yolo.py
```

## Manual Training
1. Optional: Resume Training (if you previously started training a model)
```bash
python .\start_here\train\resume_training.py
```
2. Training a baseline model (50 epochs)
```bash
python .\start_here\train\train_baseline.py
```

## Confirmed Supported Python Versions
- Python 3.10
- Python 3.11.0

## Sources
- [mju-waste Github](https://github.com/realwecan/mju-waste)
- [MJU-Waste Dataset](https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view)