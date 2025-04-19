import os
import re
import cv2
import numpy as np
from pathlib import Path

def parse_dimensions_from_filename(filename):
    """
    Extract dimensions (width x height) from filenames like: name_3128x1946.gray
    """
    match = re.search(r"(\d+)x(\d+)", filename)
    if not match:
        raise ValueError(f"Can't parse dimensions from: {filename}")
    width, height = int(match.group(1)), int(match.group(2))
    return width, height

def convert_gray_to_png_recursive(root_dir, output_dir='yolo_dataset/images/train'):
    root = Path(root_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for gray_file in root.rglob("*.gray"):
        try:
            width, height = parse_dimensions_from_filename(gray_file.name)

            with open(gray_file, 'rb') as f:
                raw = np.frombuffer(f.read(), dtype=np.uint8)
                img = raw.reshape((height, width))

            # Save as PNG in output directory
            out_filename = gray_file.stem + ".png"
            out_path = output / out_filename
            cv2.imwrite(str(out_path), img)
            print(f"[+] Converted: {gray_file.name} → {out_path.name}")

            # Create an empty label file in yolo_dataset/labels/train/
            label_dir = Path(output_dir.replace("images", "labels"))
            label_dir.mkdir(parents=True, exist_ok=True)
            label_file = label_dir / (gray_file.stem + ".txt")
            label_file.write_text("")  # Blank annotation to start with

        except Exception as e:
            print(f"[!] Error converting {gray_file.name}: {e}")

if __name__ == "__main__":
    convert_gray_to_png_recursive("images")
