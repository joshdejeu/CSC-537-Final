# ===========================================================================================
# Run this to see if you have GPU available (you may need to reinstall pytorch for older GPU)
# pip install torch
# ===========================================================================================

import torch
def main():
    print("\nTorch version: ", torch.__version__) # Verifys PyTorch installation
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

if __name__ == "__main__":
    main()


# CUDA 12.1 for 5070
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Use [this site](https://www.techpowerup.com/gpu-specs/geforce-rtx-5070.c4218) to find the cuda version required for your GPU