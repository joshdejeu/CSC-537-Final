> **Note:** Detectron2 requires **[Python 3.10](https://www.python.org/downloads/release/python-3100/)**


# macOS/Linux Setup (w/ CUDA 12.1)

#### Open a terminal and run these commands, replace the path to python.exe if needed

## 1. Install Python 3.10
```bash
brew install python@3.10                     # macOS
sudo apt install python3.10 python3.10-venv  # Linux
```
## 2. Create and Activate `Python 3.10` venv
```bash
python3.10 -m venv envs/detectron_env
source envs/detectron_env/bin/activate```
```

## 3. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

## 4. Install CPU or MPS PyTorch
- macOs
```bash
pip install torch torchvision torchaudio
```

- Linux
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 5. Install Detectron2
```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

