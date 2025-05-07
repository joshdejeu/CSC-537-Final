# ==================================================
# Downloads / Organizes MJU-waste imgs & annotations
# ==================================================

import subprocess
import sys
import os

# Gets the numpy version based on the Python version
def get_numpy_version():
    major, minor = sys.version_info[:2]
    if major == 3 and minor <= 6:
        return "numpy==1.19.5"
    elif major == 3 and minor == 7:
        return "numpy==1.21.6"
    elif major == 3 and minor == 8:
        return "numpy==1.24.4"
    else:
        return "numpy"  # For >=3.9 pip will install the latest version

def bootstrap():
    try:
        import gdown
        import tqdm # For progress bars
    except ImportError:
        print("Installing bootstrap requirements...")
        numpy_pkg = get_numpy_version()

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "gdown", numpy_pkg],
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        if result.returncode != 0:
            print(f"[!] Failed to install gdown + {numpy_pkg}")
            sys.exit(1)

        # Retry imports
        import gdown
        import tqdm

bootstrap() # Required libraries
print("\nSuccessfully installed bootstrap requirements.")

from setup import download_data, organize_dataset

def promptDownload():
    response = input("\n\nDownload dataset? [y/N]: ").strip().lower()
    if response == "y":
        # Download imgs from Google Drive
        download_data.main()

        # Organize the dataset into 'images' and 'mask' folders; with 'train', 'test', and 'val' subfolders
        organize_dataset.main()

def main():
    promptDownload() # If user wants to download data again

if __name__ == "__main__":
    main()
