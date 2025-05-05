> **Note:** Detectron2 requires **[Python 3.10](https://www.python.org/downloads/release/python-3100/)**

##### - When installing `Python 3.10` make sure to check `"Add python.exe to PATH"`
##### - Then choose `Custom intallation` > `Next` and copy the `Custom install location` path for the next step

# Windows Setup (w/ CUDA 12.1)

## 1. Create the Virtual Enviroment

#### Open a terminal and run these commands, replace the path to python.exe if needed

- If `Python 3.10` is your system default
```bash
python -m venv envs\detectron_env
```
- Otherwise, use the full path to the correct Python version
```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python310\python.exe" -m venv envs\detectron_env
```
"E:\Users\josh\AppData\Local\Programs\Python\Python310\python.exe" 
## 2. Activate the `Python 3.10` enviroment
```bash
.\envs\detectron_env\Scripts\activate
```
> **Note:** To leave venv, type `deactivate` in the terminal

## 4. Upgrade base tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 5. Install Build Tools (only needed once)

Make sure you have:

- [Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **"Desktop development with C++"** workload installed
- `cmake` and `ninja` Python packages installed:
```bash
pip install cmake ninja
```

## 6. Install PyTorch w/ CUDA 12.1 (2.4 GB)
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```
pip install "git+https://github.com/facebookresearch/detectron2.git#egg=detectron2" --no-cache-dir


## Install Detectron2
```bash
pip install git+https://github.com/facebookresearch/detectron2.git

```
