#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/local/models"
vendor="$root/local/vendor/rknn_model_zoo"
revision=837db8e992815ff8647eadb709f07067f4bc3a71
selection="${1:-all}"
case "$selection" in
  all|yolo26m|ppocrv4) ;;
  *) echo "Usage: bash tools/compile_models.sh [all|yolo26m|ppocrv4]" >&2; exit 2 ;;
esac

if [[ ! -d "$vendor" ]]; then
  mkdir -p "$root/local/vendor"
  (
    staging="$(mktemp -d "$root/local/vendor/.model-zoo.XXXXXX")"
    trap 'rm -rf -- "$staging"' EXIT
    git init "$staging"
    git -C "$staging" fetch --depth 1 https://github.com/airockchip/rknn_model_zoo.git "$revision"
    git -C "$staging" checkout --detach FETCH_HEAD
    mv -nT -- "$staging" "$vendor"
  )
fi

converter_python="${RKNN_PYTHON:-$root/venv/bin/python}"
if [[ ! -x "$converter_python" ]]; then
  echo "Set up venv and install requirements.txt before compiling models." >&2
  exit 1
fi
converter_python="$(realpath -s -- "$converter_python")"
"$converter_python" -c 'from importlib.metadata import version; from rknn.api import RKNN; assert version("rknn-toolkit2") == "2.5.0", "Expected RKNN-Toolkit2 2.5.0"'

compile() {
  local example="$1" source="$models/$2" target="$models/$3"
  if [[ -s "$target" ]]; then
    printf 'Using %s\n' "$target"
    return
  fi
  if [[ ! -s "$source" ]]; then
    printf 'Missing %s. Run bash tools/download_models.sh %s first.\n' "$source" "$selection" >&2
    return 1
  fi
  (
    staging="$(mktemp -d "$(dirname "$target")/.compile.XXXXXX")"
    trap 'rm -rf -- "$staging"' EXIT
    cd "$vendor/examples/$example/python"
    "$converter_python" convert.py "$source" rk3588 fp "$staging/model.rknn"
    test -s "$staging/model.rknn"
    mv -n -- "$staging/model.rknn" "$target"
  )
}

if [[ "$selection" == all || "$selection" == yolo26m ]]; then
  compile yolo26 yolo26m.onnx yolo26m-rk3588-fp16.rknn
fi
if [[ "$selection" == all || "$selection" == ppocrv4 ]]; then
  compile PPOCR/PPOCR-Det ppocrv4/ppocrv4_det.onnx ppocrv4/ppocrv4_det.rknn
  compile PPOCR/PPOCR-Rec ppocrv4/ppocrv4_rec.onnx ppocrv4/ppocrv4_rec.rknn
fi
