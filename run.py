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

        # Split dataset into (train, test, val)
        organize_dataset.main()

def main():
    promptDownload() # If user wants to download data again

if __name__ == "__main__":
    main()
