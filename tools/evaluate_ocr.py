"""Compare RKNN PP-OCRv4 crop predictions with verified annotations."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2

from models.ppocrv4 import RKNNTextReader


def evaluate(annotations, images_dir, output):
    dataset = json.loads(annotations.read_text(encoding="utf-8"))
    dimensions = {item["image"]: (item["width"], item["height"]) for item in dataset["images"]}
    by_image = defaultdict(list)
    for row in dataset["readings"]:
        if row["status"] == "verified":
            by_image[row["image"]].append(row)
    if not by_image:
        raise ValueError("No verified readings in annotations")

    results = []
    reader = RKNNTextReader(crop_only=True)
    try:
        for name, readings in by_image.items():
            frame = cv2.imread(str(images_dir / name))
            if frame is None:
                raise ValueError(f"Could not read annotated image: {name}")
            height, width = frame.shape[:2]
            if (width, height) != dimensions[name]:
                raise ValueError(f"Image dimensions differ from annotations: {name}")
            for row in readings:
                x1, y1, x2, y2 = (row[key] for key in ("x1", "y1", "x2", "y2"))
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Invalid crop coordinates for {row['id']} in {name}")
                start = perf_counter()
                prediction, confidence = reader.recognize_crop(frame[y1:y2, x1:x2])
                elapsed_ms = (perf_counter() - start) * 1000
                results.append({
                    "id": row["id"], "image": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "expected": row["value"], "predicted": prediction,
                    "exact_match": prediction == row["value"],
                    "model_confidence": round(float(confidence), 4),
                    "latency_ms": round(elapsed_ms, 2),
                })
    finally:
        reader.close()

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    matches = sum(row["exact_match"] for row in results)
    blanks = sum(not row["predicted"] for row in results)
    count = len(results)
    print(f"{count} verified crops across {len(by_image)} images")
    print(f"{matches} exact matches ({matches / count:.1%}), "
          f"{count - matches - blanks} nonempty mismatches, {blanks} blank predictions")
    print(f"Average crop processing: {sum(row['latency_ms'] for row in results) / count:.1f} ms")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.annotations, args.images, args.output)


if __name__ == "__main__":
    main()
