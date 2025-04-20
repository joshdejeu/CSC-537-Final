
> **Note:** Matterport requires **[Python 3.6.8](https://www.python.org/downloads/release/python-368/)**, avoid Python 3.6.0 since it has outdated SSL root certs

##### - When installing `Python 3.6.8` make sure to check `"Add python.exe to PATH"`
##### - Then choose `Custom intallation` > `Next` and copy the `Custom install location` path for the next step

## 1. Create the Virtual Enviroment

#### Open a terminal and run these commands, replace the path to python.exe if needed

- If `Python 3.6.8` is your system default
```bash
python -m venv envs\matterport_env
```
- Otherwise, use the full path to the correct Python version
```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python38\python.exe" -m venv envs\matterport_env
```

## 2. Activate the `Python 3.6.8` enviroment
```bash
.\envs\matterport_env\Scripts\activate
```
> **Note:** To leave venv, type `deactivate` in the terminal

## 3. Verify that you're using `Python 3.6.8`
```bash
python --version
```

### ⚠️ Upgrade pip, setuptools, and wheel before installing requirements to avoid PEP 517 build errors
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Matterport requirements
```bash
pip install -r .\models\matterport\requirements.txt
```

## 5. Traing the model (CPU default if no GPU present)
```bash
python .\models\matterport\train\train.py
```

### Optional: Resume training if weights have been saved after 1 epoch
```bash
python .\models\matterport\train\resume_training.py
```

## 6. Visualize training progression
```bash
tensorboard --logdir logs
```

## 7. Then go to [http://localhost:6006/](http://localhost:6006/)

## 8. Use your trained weights to predict images
```bash
python .\models\matterport\predict\predict.py
```

## CUDA downloads for GPU training 
- [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [cuDNN 7.4](https://developer.nvidia.com/rdp/cudnn-archive)
> **Note:** Make sure to extract the cuDNN files into your CUDA installation directory (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0`)