import os
import requests
from pathlib import Path

# روابط الموديلات من Google Drive
models = {
    "Car_Brand/best.pt": "https://huggingface.co/Waled2002/car-detection-models/resolve/main/Car_Brand/best.pt",
    "Car_Color/best.pt": "https://huggingface.co/Waled2002/car-detection-models/resolve/main/Car_Color/best.pt",
    "Car_Plate/best.pt": "https://huggingface.co/Waled2002/car-detection-models/resolve/main/Car_Plate/best.pt",
}

# دالة لتحميل الموديلات من Google Drive
def download_model(path, url):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {path}")
    else:
        print(f"{path} already exists.")

# تحميل كل الموديلات
if __name__ == "__main__":
    for path, url in models.items():
        download_model(path, url)
