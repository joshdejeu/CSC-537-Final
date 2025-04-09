
import gdown
import zipfile
import os

google_drive_url = "https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view?usp=sharing"
tmp_output_zip = "tmp/mju-waste.zip"
output_dir = "tmp"

os.makedirs(output_dir, exist_ok=True) # Check if output folder exists

# Download with fuzzy=True to handle full Google Drive share link
if not os.path.exists(tmp_output_zip):
    print("Downloading dataset...")
    gdown.download(url=google_drive_url, output=tmp_output_zip, quiet=False, fuzzy=True)

# Now unzip
if os.path.exists(tmp_output_zip):
    print("Extracting images...")
    with zipfile.ZipFile(tmp_output_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir) # Extract zip into 'output_dir' folder

    os.remove(tmp_output_zip)
    print("Dataset downloaded and extracted.")
else:
    print("Download failed. Zip file not found.")
