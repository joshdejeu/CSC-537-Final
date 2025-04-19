#### **Note:** YOLO requires **[Python 3.8](https://www.python.org/downloads/release/python-380/)**

##### - When installing `Python 3.8` make sure to check `"Add python.exe to PATH"`

##### - Then choose `Custom intallation` > `Next` and copy the `Custom install location` path for the next step

## 1. Create the Virtual Enviroment

#### Open a terminal and run these commands, replace the path to python.exe if needed

- If `Python 3.8` is your system default
```bash
python -m venv envs\yolo_env
```
- Otherwise, use the full path to the correct Python version
```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python38\python.exe" -m venv envs\yolo_env
```

## 2. Activate the `Python 3.8` enviroment
```bash
.\envs\yolo_env\Scripts\activate
```
**Note:** To leave venv, type `deactivate` in the terminal

## 3. Verify that you're using `Python 3.8`
```bash
python --version
```

### ⚠️ Upgrade pip, setuptools, and wheel before installing requirements to avoid PEP 517 build errors
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Ultralytics requirements in venv (auto-detects GPU support)
```bash
python .\models\yolo\setup\install_yolo_requirements.py
```

## 5. Translate COCO-style annotations into YOLOv8 format
```bash
python .\models\yolo\setup\convert_coco_to_yolo.py
```

## 6. Train your YOLO model
```bash
python .\models\yolo\train\train.py
```

### Optional: Resume training if weights have been saved after 1 epoch
```bash
python .\models\yolo\train\resume_training.py
```

## 7. Use your trained weights to predict images
```bash
python .\models\yolo\predict\predict.py
```
