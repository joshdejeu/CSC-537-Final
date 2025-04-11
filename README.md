## Set Up
- Make sure to download requirements beforehand
```bash
pip install -r requirements.txt
```

### 1. Download the Dataset
```bash
python .\start_here\A_setup\download_data.py
```
### 2. Organize Images to Respective Folders
```bash
python .\start_here\A_setup\organize_dataset.py
```
### 3. Convert COCO annotations to YOLOv8 format
```bash
python [TODO]
```

## Training
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