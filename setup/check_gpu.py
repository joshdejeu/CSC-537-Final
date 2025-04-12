# Run this to see if you have GPU available (you may need to reinstall pytorch for older GPU)
import torch
def main():
    print("\nTorch version: ", torch.__version__) # Verifys PyTorch installation
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

if __name__ == "__main__":
    main()