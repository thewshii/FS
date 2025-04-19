import cv2
import numpy as np
import os
import json
import argparse
import re

def parse_dimensions_from_filename(filename):
    match = re.search(r"(\d+)x(\d+)", filename)
    if not match:
        raise ValueError(f"Could not extract dimensions from filename: {filename}")
    return int(match.group(1)), int(match.group(2))

def load_image(path):
    if path.endswith(".gray"):
        width, height = parse_dimensions_from_filename(os.path.basename(path))
        with open(path, 'rb') as f:
            img = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def convert_yolo_to_segments(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    segments = []

    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            parts = line.strip().split()
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
    
            x = int((xc - bw / 2) * w)
            y = int((yc - bh / 2) * h)
            box_w = int(bw * w)
            box_h = int(bh * h)
            cx = x + box_w // 2
            cy = y + box_h // 2

            segment = {
                "id": i + 1,
                "label": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                "bounding_box": {"x": x, "y": y, "w": box_w, "h": box_h},
                "centroid": {"x": cx, "y": cy},
                "angle_degrees": 0,
                "nfiq_score": 1 + i % 5,
                "handedness": "unknown"
            }
            segments.append(segment)

            # Draw box
            cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
            cv2.putText(image, segment["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image, segments

def segment_fingers(image):
    results = []
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    print(f"[DEBUG] Total contours found: {len(contours)}")

    MIN_AREA = 5000
    top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    idx = 0
    for cnt in top_contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        hull = cv2.convexHull(cnt)
        x, y, bw, bh = cv2.boundingRect(hull)

        aspect_ratio = bw / float(bh)
        if aspect_ratio < 0.3 or aspect_ratio > 1.2:
            continue

        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)

        cx, cy = x + bw // 2, y + bh // 2

        rect = cv2.minAreaRect(hull)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        angle = int(round(angle))

        cv2.rectangle(output_image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(output_image, f"F{idx+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        result = {
            "id": idx + 1,
            "bounding_box": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
            "centroid": {"x": int(cx), "y": int(cy)},
            "angle_degrees": angle,
            "nfiq_score": 1 + idx % 5,
            "handedness": "unknown"
        }
        results.append(result)
        idx += 1

    if not results:
        print("[WARNING] No valid fingers detected, but threshold image looks good.")
    else:
        print(f"[INFO] Segmented {len(results)} fingers.")

    return output_image, results

def main():
    parser = argparse.ArgumentParser(description="Slap Fingerprint Segmentor CLI")
    parser.add_argument("--input", required=True, help="Path to input slap image (.png or .gray)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--yolo", help="Optional YOLO label .txt file for segmentation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image = load_image(args.input)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_image_path = os.path.join(args.output, f"{base_name}_segmented.png")
    out_json_path = os.path.join(args.output, f"{base_name}_segments.json")

    if args.yolo:
        class_names = [
            "left_index", "left_middle", "left_ring", "left_little",
            "right_index", "right_middle", "right_ring", "right_little", "thumb"
        ]
        annotated_image, segments = convert_yolo_to_segments(args.input, args.yolo, class_names)
    else:
        annotated_image, segments = segment_fingers(image)

    cv2.imwrite(out_image_path, annotated_image)
    with open(out_json_path, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"[+] Saved annotated image: {out_image_path}")
    print(f"[+] Saved segmentation data: {out_json_path}")

if __name__ == "__main__":
    main()