"""Run a registered vision model on the RK3588 NPU."""
import argparse
import os
from pathlib import Path
from time import monotonic

import cv2

from models.ppocrv4 import RKNNTextReader
from models.yolo26m import YOLODetector

MODELS = {
    "detect": {"yolo26m": YOLODetector},
    "ocr": {"ppocrv4": RKNNTextReader},
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def run_images(process, paths):
    start = monotonic()
    results = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        image_start = monotonic()
        result = process(frame)
        results.append((path, result, monotonic() - image_start))
    elapsed = monotonic() - start
    for path, result, seconds in results:
        print(f"=== {path.name} | {seconds * 1000:.1f} ms ===")
        print(result or "[No results]")
    print(f"{len(paths)} images | {len(paths) / elapsed:.1f} FPS (file read and processing)")


def run_camera(process, task):
    device = "/dev/video0"
    preview = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        previous_text = None
        start = monotonic()
        count = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Cannot read a frame from {device}")
            if task == "detect":
                result = process(frame, preview=preview)
            else:
                result = process(frame)
                if result and result != previous_text:
                    print(result, flush=True)
                previous_text = result
            if preview:
                cv2.imshow(task.upper(), frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            count += 1
            now = monotonic()
            if now - start >= 2:
                status = f"{count / (now - start):.1f} camera FPS"
                if task == "detect" and not preview:
                    status += f" | {len(result.splitlines())} detections in latest frame"
                print(status, flush=True)
                start, count = now, 0
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODELS, help="Vision task")
    model_help = "; ".join(f"{task}: {', '.join(models)}" for task, models in MODELS.items())
    parser.add_argument("model", help=f"Model name ({model_help})")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("path", type=Path, nargs="?", help="Path to an image or image directory")
    source.add_argument("--camera", action="store_true", help="Read the USB camera at /dev/video0")
    args = parser.parse_args()

    available = MODELS[args.task]
    if args.model not in available:
        parser.error(f"Models for {args.task}: {', '.join(available)}")

    paths = []
    if args.path is not None:
        if args.path.is_file():
            paths = [args.path]
        elif args.path.is_dir():
            paths = sorted(path for path in args.path.iterdir()
                           if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if not paths:
                parser.error(f"No supported images in {args.path}")
        else:
            parser.error(f"Image or directory does not exist: {args.path}")

    process = available[args.model]()

    try:
        if args.camera:
            run_camera(process, args.task)
        else:
            run_images(process, paths)
    except KeyboardInterrupt:
        pass
    finally:
        process.close()


if __name__ == "__main__":
    main()
