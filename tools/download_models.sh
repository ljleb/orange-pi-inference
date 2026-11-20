#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/local/models"
selection="${1:-all}"
case "$selection" in
  all|yolo26m|ppocrv4) ;;
  *) echo "Usage: bash tools/download_models.sh [all|yolo26m|ppocrv4]" >&2; exit 2 ;;
esac

mkdir -p "$models"
staging="$(mktemp -d "$models/.download.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT

download_ocr() {
  local name="$1" checksum="$2" target="$models/ppocrv4/$1"
  local base="https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/PPOCR"
  mkdir -p "$models/ppocrv4"
  if [[ -e "$target" ]]; then
    printf '%s  %s\n' "$checksum" "$target" | sha256sum --check --status
    printf 'Using %s\n' "$target"
    return
  fi
  curl --fail --location --retry 2 --output "$staging/$name" "$base/$name"
  printf '%s  %s\n' "$checksum" "$staging/$name" | sha256sum --check
  mv -n -- "$staging/$name" "$target"
}

if [[ "$selection" == all || "$selection" == ppocrv4 ]]; then
  download_ocr ppocrv4_det.onnx 55a2092c7068bc8e9c79e4564958beaacdd66e802fd2eeffbba7604003dbb5c2
  download_ocr ppocrv4_rec.onnx 773fc4e13c25f4847fb82aa2112dc673355c13c0b32b066ae2fb082478810916
fi

if [[ "$selection" == all || "$selection" == yolo26m ]]; then
  if [[ -s "$models/yolo26m.onnx" ]]; then
    printf 'Using %s\n' "$models/yolo26m.onnx"
    exit 0
  fi
  export_python="$root/venv/bin/python"
  if [[ ! -x "$export_python" ]]; then
    echo "Set up venv and install requirements.txt before exporting YOLO26m." >&2
    exit 1
  fi
  weights="$models/yolo26m.pt"
  weights_checksum=401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7
  if [[ ! -s "$weights" ]]; then
    curl --fail --location --retry 2 --output "$staging/yolo26m.pt" \
      'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt?download=1'
    printf '%s  %s\n' "$weights_checksum" "$staging/yolo26m.pt" | sha256sum --check
    mv -n -- "$staging/yolo26m.pt" "$weights"
  else
    printf '%s  %s\n' "$weights_checksum" "$weights" | sha256sum --check
  fi
  (
    cd "$staging"
    YOLO_CONFIG_DIR="$root/local/ultralytics" "$export_python" - "$weights" <<'PYCODE'
import shutil
import sys

import onnx
import torch
from ultralytics import YOLO

torch.set_num_threads(2)
shutil.copyfile(sys.argv[1], "yolo26m.pt")
path = YOLO("yolo26m.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False,
                                  simplify=True, nms=False, device="cpu")
graph = onnx.load(path).graph
required = {"/model.23/Sigmoid_output_0", "/model.23/Mul_2_output_0"}
missing = required - {name for node in graph.node for name in node.output}
if missing:
    raise RuntimeError(f"ONNX export is missing Rockchip converter outputs: {sorted(missing)}")
PYCODE
  )
  mv -n -- "$staging/yolo26m.onnx" "$models/yolo26m.onnx"
fi
