#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/venv/bin/python"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT

"$python" - "$scratch/smoke.png" <<'PYCODE'
import sys

import cv2
import numpy as np

image = np.full((480, 480, 3), 255, dtype=np.uint8)
cv2.putText(image, "TEST 123", (35, 250), cv2.FONT_HERSHEY_SIMPLEX,
            2, (0, 0, 0), 4, cv2.LINE_AA)
if not cv2.imwrite(sys.argv[1], image):
    raise SystemExit("Could not write the smoke-test image")
PYCODE

"$python" "$root/rknn_vision.py" detect yolo26m "$scratch/smoke.png"
"$python" "$root/rknn_vision.py" ocr ppocrv4 "$scratch/smoke.png" | tee "$scratch/ocr.txt"
grep -Fxq 'TEST 123' "$scratch/ocr.txt"
echo "NPU smoke test passed."
