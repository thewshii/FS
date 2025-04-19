import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
import re

YOLO_MODEL = "runs/detect/train/weights/best.pt"
SEGMENTOR_SCRIPT = "segmentor.py"
GRAY_DIR = Path("images")
OUTPUT_DIR = Path("output")
PREDICT_TMP = Path("runs/detect/predict")

def parse_gray_dimensions(filename):
    match = re.search(r"(\d+)x(\d+)", filename)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

def convert_gray_to_png(gray_path):
    width, height = parse_gray_dimensions(gray_path.name)
    if width == 0 or height == 0:
        raise ValueError(f"[!] Could not parse dimensions from {gray_path.name}")

    # Load and reshape grayscale raw
    with open(gray_path, 'rb') as f:
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
    try:
        img = img_data.reshape((height, width))
    except ValueError:
        raise ValueError(f"[!] Reshape failed for {gray_path.name} with dimensions ({width}, {height})")

    # Normalize + convert to RGB
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    png_path = gray_path.with_suffix('.png')
    cv2.imwrite(str(png_path), img_rgb)
    return png_path

def run_yolo_inference(png_path):
    try:
        subprocess.run([
    "yolo", "detect", "predict",
    f"model={YOLO_MODEL}",
    f"source={str(png_path)}",
    "conf=0.01",
    "imgsz=1280",
    "save=True",
    "project=runs/detect",
    "name=predict",
    "exist_ok=True"
], check=True)

    except subprocess.CalledProcessError as e:
        print(f"[!] YOLO failed on {png_path.name}: {e}")
        return False
    return True

def run_segmentor(gray_file):
    image_name = gray_file.name.replace(".gray", ".png")
    input_image = PREDICT_TMP / image_name
    label_file = PREDICT_TMP / "labels" / image_name.replace(".png", ".txt")

    if not input_image.exists() or not label_file.exists():
        print(f"[!] Missing YOLO output for {gray_file.name}")
        return

    subprocess.run([
        "python", SEGMENTOR_SCRIPT,
        "--input", str(input_image),
        "--yolo", str(label_file),
        "--output", str(OUTPUT_DIR)
    ])

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    for gray_file in GRAY_DIR.glob("*.gray"):
        print(f"[+] Predicting on {gray_file.name}")
        try:
            png_path = convert_gray_to_png(gray_file)
        except Exception as e:
            print(f"[!] Skipping {gray_file.name}: {e}")
            continue

        if run_yolo_inference(png_path):
            run_segmentor(gray_file)

if __name__ == "__main__":
    main()
