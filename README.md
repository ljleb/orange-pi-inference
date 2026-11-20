# RKNN Vision

Run YOLO26m object detection and PP-OCRv4 text recognition on a Rockchip RK3588. The CLI accepts an image, a directory of images, or a live USB camera.

## Setup

Use 64-bit Ubuntu 22.04 or 24.04 on the Orange Pi with an RK3588-compatible kernel driver. Run these commands from the repository root. The same virtual environment handles export, compilation, and inference.

```bash
sudo apt update
sudo apt install -y git curl python3-venv libgl1 libglib2.0-0
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip check
```

Install Rockchip's [ARM64 RKNN runtime 2.5.0](https://github.com/airockchip/rknn-toolkit2/blob/v2.5.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so) if the board doesn't already have that version:

```bash
mkdir -p local/lib
curl -fL https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.5.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
  -o local/lib/librknnrt.so
sudo install -m 0644 local/lib/librknnrt.so /usr/lib/librknnrt.so
```

## Download and compile the models

```bash
bash tools/download_models.sh
bash tools/compile_models.sh
bash tools/smoke_test.sh
```

Both scripts accept `yolo26m` or `ppocrv4` to prepare just one model. They reuse files already present under `local/models/`. To rebuild a model, move its existing `.rknn` file aside first.

The download script verifies the SHA-256 hashes of Rockchip's PP-OCRv4 ONNX files. For YOLO26m, it exports the official Ultralytics weights to a 640×640 ONNX model using PyTorch 2.2.0, the version pinned for compatibility with RKNN-Toolkit2 2.5.0. You can also use Rockchip's [YOLO26m ONNX export](https://meta.zbox.filez.com/v/link/view/198e90ee6bc94c71ad2002d5f9897ac4) at `local/models/yolo26m.onnx`.

The compile script fetches a pinned Rockchip Model Zoo revision when the vendor checkout is absent. The checkout also supplies PP-OCR postprocessing and its character dictionary. The smoke test runs both models on a temporary synthetic image and checks that OCR reads its text. No camera or dataset is needed.

| CLI task and model | Local model files | Conversion guide |
| --- | --- | --- |
| `detect yolo26m` | `local/models/yolo26m-rk3588-fp16.rknn` | [YOLO26](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo26) |
| `ocr ppocrv4` | `local/models/ppocrv4/ppocrv4_det.rknn` and `local/models/ppocrv4/ppocrv4_rec.rknn` | [PP-OCRv4](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR/PPOCR-System) |

## Run

```bash
venv/bin/python rknn_vision.py detect yolo26m /path/to/image.png
venv/bin/python rknn_vision.py ocr ppocrv4 /path/to/images
venv/bin/python rknn_vision.py detect yolo26m --camera
```

Run the commands from the repository root. The CLI selects a model by name and detects whether the input path is a file or directory. Camera mode uses `/dev/video0` and takes no path.

## Evaluate OCR on annotated crops

Supply your own screenshots and annotations. The JSON contains an `images` array with `image`, `width`, and `height`, and a `readings` array with `id`, `image`, `value`, `status`, and the crop coordinates `x1`, `y1`, `x2`, `y2`. Coordinates are in source-image pixels, with exclusive right and bottom edges. The evaluator scores rows marked `verified` by exact text match.

```bash
mkdir -p local/results
venv/bin/python -m tools.evaluate_ocr \
  local/data/annotations.json local/data/images local/results/ocr_comparison.csv
```

Keep screenshots and annotations under `local/data/` and generated reports under `local/results/`. All of `local/` is ignored by Git.
