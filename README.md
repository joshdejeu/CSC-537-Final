# CSC 537 Final Project
#### Deep learning-based image segmentation for garbage detection

## Supported Python Versions
- Python 3.8.x (YOLOv8)
- Python 3.6.8 (Matterport)

## Folder Structure
```
CSC-537-Final
 ┣ datasets                 # Contains dataset files and configurations
 ┃ ┣ COCO_annotations       # COCO-style annotations (train, val, test) from official MJU-waste github
 ┃ ┣ images                 # Original PNG images (train, val, test)
 ┃ ┣ labels                 # YOLOv8 format labels derived from COCO annotations
 ┃ ┣ masks                  # Pre-generated segmentation masks (train, val, test)
 ┃ ┗ mju.yaml               # Config file for YOLOv8
 ┣ envs                     # Virtual environments for different models
 ┃ ┣ matterport_env
 ┃ ┗ yolo_env
 ┣ models                   # Model-specific setup and training scripts
 ┃ ┣ matterport
 ┃ ┗ yolo
 ┣ runs                     # All YOLO training runs will be in segment/
 ┃ ┗ segment
 ┣ setup                    # General setup that all models share
 ┣ tmp                      # Temporary storage (when downloading img .zip)
 ┗ download.py              # Download and organize data, install bootstrap requirements
```

<br>

# Download dataset
- Download bootstrap requirements & image dataset (which automatically gets split into folders)
```bash
python .\download.py
```

<br>

# Ultralytics YOLOv8
1. Create virtual enviroment for model with Python 3.8 (64 bit)
#### What is `python.exe`? If Python 3.8 is your default you can use `python` instead, else use `C:\Users\<user>\AppData\Local\Programs\Python\Python38\python.exe`
```bash
python.exe -m venv envs\yolo_env
.\envs\yolo_env\Scripts\activate
```
**Note:** To leave venv, type `deactivate` in the terminal
2. Double check the Python version, should say `Python 3.6.8`
```bash
python --version
```
### ⚠️ Upgrade pip, setuptools, and wheel before installing requirements to avoid PEP 517 build errors
```bash
python -m pip install --upgrade pip setuptools wheel
```
3. Install Ultralytics requirements in venv (auto-detects GPU support)
```bash
python .\models\yolo\setup\install_yolo_requirements.py
```
4. Translate COCO-style annotations into YOLOv8 format
```bash
python .\models\yolo\setup\convert_coco_to_yolo.py
```
5. Train your YOLO model
```bash
python .\models\yolo\train\train.py
```
6. Optional: Resume training if weights have been saved after 1 epoch
```bash
python .\models\yolo\train\resume_training.py
```

<br>



# Matterport Mask R-CNN
**Note:** Use [Python 3.6.8](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe) (3.6.0 has outdated SSL root certs)
1. Create virtual enviroment for model with Python 3.6.8 (64 bit)
#### What is `python.exe`? If Python 3.6.8 is your default you can use `python` instead, else use `C:\Users\<user>\AppData\Local\Programs\Python\Python38\python.exe`
```bash
python.exe -m venv envs\matterport_env
.\envs\matterport_env\Scripts\activate
```
**Note:** To leave venv, type `deactivate` in the terminal
2. Double check the Python version, should say `Python 3.6.8`
```bash
python --version
```
### ⚠️ Upgrade pip, setuptools, and wheel before installing requirements to avoid PEP 517 build errors
```bash
python -m pip install --upgrade pip setuptools wheel
```
3. Install Matterport requirements
```bash
pip install -r .\models\matterport\requirements.txt
```
4. TODO: Traing the model
```bash
python .\models\matterport\train\train.py
```
5. Visualize log file
```bash
tensorboard --logdir logs
```
6. Then go to `http://localhost:6006/`

<br>

## Authors
- Josh Dejeu
- Aniya Watson
- Abhinav Medarametla

## Sources
- [mju-waste Github](https://github.com/realwecan/mju-waste)
- [MJU-Waste Dataset](https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view)
- [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md)
- [Matterport](https://github.com/matterport/Mask_RCNN)