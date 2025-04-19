# create_mock_slap.py
# Utility script to build a mock slap image from FVC .tif files in a ZIP

import os
import zipfile
import cv2
import numpy as np
import tempfile
import sys


def extract_tifs_from_zip(zip_path, limit=4):
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        tifs = [f for f in zip_ref.namelist() if f.endswith('.tif')]
        for i in range(min(limit, len(tifs))):
            with zip_ref.open(tifs[i]) as file:
                temp_path = os.path.join(tempfile.gettempdir(), f"finger_{i}.tif")
                with open(temp_path, 'wb') as out:
                    out.write(file.read())
                extracted_files.append(temp_path)
    return extracted_files


def build_mock_slap(finger_paths, size=(300, 300)):
    fingers = []
    for path in finger_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, size)
        fingers.append(resized)

    if len(fingers) < 4:
        raise Exception("Less than 4 usable fingerprint images found.")

    slap = np.hstack(fingers)
    out_path = "mock_slap.png"
    cv2.imwrite(out_path, slap)
    print(f"[+] Mock slap image created: {out_path}")
    return out_path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python create_mock_slap.py DB1_B.zip")
        sys.exit(1)

    zip_path = sys.argv[1]
    tifs = extract_tifs_from_zip(zip_path)
    build_mock_slap(tifs)
