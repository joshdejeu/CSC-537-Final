# Matterport Mask R-CNN Setup (macOS Version)

> **Note:** Matterport requires **[Python 3.6.8](https://www.python.org/downloads/release/python-368/)**, avoid Python 3.6.0 since it has outdated SSL root certs
\
> Direct download: [macOS 64-bit installer (.pkg)](https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg)

## 0. Install Python 3.6.8 (macOS)
```swift
/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
```

## 1. Create a virtual env using the install path from the last step
```swift
/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -m venv envs/matterport_env
```

## 2. Activate the virtual enviroment
```bash
source envs/matterport_env/bin/activate
```

## 3. Double check the Python version, should be `Python 3.6.8`
```bash
python --version
```

### ⚠️ Important: Upgrade pip, setuptools, and wheel before installing requirements to avoid PEP 517 build errors
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Matterport Requirements
```bash
pip install -r ./models/matterport/requirements.txt
```

## 5. Traing the model (CPU only on Mac)
```bash
python ./models/matterport/train/train.py
```

### Optional: Resume training on a prev run (if weights have been saved after 1 epoch)
```bash
python ./models/matterport/train/resume_training.py
```

## 6. Visualize training progression
> Note: This can be done even after training, it uses the event file that's saved along the weight files
```bash
tensorboard --logdir output/mrcnn
```
Then go to [http://localhost:6006/](http://localhost:6006/)

## 7. Use your trained weights to predict images
```bash
python .\models\matterport\predict\predict_custom.py
```
