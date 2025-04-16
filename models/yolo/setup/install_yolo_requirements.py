# ========================================================================
#  Checks if CUDA is available and installs the requirements for YOLOv8.
#  If CUDA is available, installs with CUDA 11.8 support.
#  Otherwise, installs CPU-only versions of torch, torchvision, and torchaudio.
# ========================================================================

import subprocess
import sys
import shutil

# Runs "nvidia-smi" commands to check if CUDA is available
def is_cuda_available():
    try:
        result = shutil.which("nvidia-smi")
        if result:
            print("CUDA is available (nvidia-smi detected).")
            return True
        else:
            print("CUDA not available (nvidia-smi not found).")
            return False
    except Exception as e:
        print("Error while checking for CUDA:", e)
        return False

def install_requirements(cuda_available):
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "./models/yolo/requirements.txt"], check=True)
    
    if cuda_available:
        print("CUDA is available. Installing with CUDA 11.8 support...")
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.1.0+cu118", 
            "torchvision==0.16.0+cu118", 
            "torchaudio==2.1.0+cu118",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        print("CUDA not available. Installing CPU-only version...")
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.1.0",
            "torchvision==0.16.0",
            "torchaudio==2.1.0"
        ]
        
    subprocess.run(torch_cmd, check=True)

def main():
    try:
        cuda = is_cuda_available()
        install_requirements(cuda)
    except Exception as e:
        print("Something went wrong during installation:", e)

if __name__ == "__main__":
    main()
