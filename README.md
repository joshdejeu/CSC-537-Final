## Set Up
- First, install the required packages
```bash
pip install -r requirements.txt
```
- Optional: Check if PyTorch can access the GPU (CUDA, faster than CPU)
```bash
python .\start_here\A_setup\check_gpu.py
```

### 1. Download the dataset
```bash
python .\start_here\A_setup\download_data.py
```
### 2. Organize images into their respective folders
```bash
python .\start_here\A_setup\organize_dataset.py
```
### 3. Convert COCO-style annotations to YOLOv8 format
```bash
python .\start_here\A_setup\convert_coco_to_yolo.py
```

## Training
### Optional - Resume Training (if you previously started training a model)
```bash
python .\start_here\B_train\resume_training.py
```
### 1. Training a baseline model (50 epochs)
```bash
python .\start_here\B_train\train_baseline.py
```

## Confirmed Supported Python Versions
- Python 3.10
- Python 3.11.0

## Sources
- [mju-waste Github](https://github.com/realwecan/mju-waste)
- [MJU-Waste Dataset](https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view)