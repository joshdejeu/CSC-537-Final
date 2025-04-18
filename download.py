# ===========================
# Downloads / Organizes MJU-waste imgs & annotations
# ===========================

import subprocess
import sys

def bootstrap():
    try:
        import gdown
        import tqdm # For progress bars
    except ImportError:
        print("Installing bootstrap requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", r".\setup\bootstrap_requirements.txt"])

bootstrap() # Required libraries

from setup import download_data, organize_dataset

def promptDownload():
    response = input("Download dataset? [y/N]: ").strip().lower()
    if response == "y":
        # Download imgs from Google Drive
        download_data.main()

        # Organize the dataset into 'images' and 'mask' folders; with 'train', 'test', and 'val' subfolders
        organize_dataset.main()

def main():
    promptDownload() # If user wants to download data again

if __name__ == "__main__":
    main()
