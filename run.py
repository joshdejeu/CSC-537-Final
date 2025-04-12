# ===========================
# Downloads / Organizes imags & annotations
# Choose model to train
# ===========================

from setup import download_data, organize_dataset, check_gpu

def bootstrap():
    try:
        import gdown
        import tqdm
    except ImportError:
        print("Installing bootstrap requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "setup/bootstrap_requirements.txt"])

def promptDownload():
    response = input("Download dataset? [y/N]: ").strip().lower()
    if response == "y":
        # Download imgs from Google Drive
        download_data.main()

        # Split dataset into (train, test, val)
        organize_dataset.main()

def main():
    bootstrap() # Required libraries
    promptDownload() # If user wants to download data again
    check_gpu.main() # Output GPU details

if __name__ == "__main__":
    main()
