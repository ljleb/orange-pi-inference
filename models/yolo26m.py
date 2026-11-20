"""YOLO26m object detection on the RK3588 NPU."""
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "local" / "models" / "yolo26m-rk3588-fp16.rknn"
SIZE = 640
CLASSES = (
    "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|"
    "fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|"
    "elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|"
    "skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|"
    "surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|"
    "apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|"
    "potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|"
    "cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|"
    "scissors|teddy bear|hair drier|toothbrush"
).split("|")


def prepare(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, SIZE / max(height, width))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (SIZE - resized_width) // 2
    top = (SIZE - resized_height) // 2
    image = cv2.resize(frame, (resized_width, resized_height))
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = image
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb[None], (left, top, resized_width / width, resized_height / height)


def decode(outputs, confidence, transform, frame_shape):
    if outputs is None or len(outputs) != 2:
        raise RuntimeError("Expected two outputs from the Rockchip YOLO26 converter")
    tensors = {np.asarray(output).shape: np.asarray(output) for output in outputs}
    try:
        scores = tensors[(1, len(CLASSES), 8400)][0]
        boxes = tensors[(1, 4, 8400)][0]
    except KeyError as error:
        raise RuntimeError(f"Unexpected YOLO26 output shapes: {list(tensors)}") from error
    class_ids = scores.argmax(axis=0)
    best_scores = scores[class_ids, np.arange(scores.shape[1])]
    selected = np.flatnonzero(best_scores >= confidence)
    selected = selected[np.argsort(best_scores[selected])[::-1][:300]]

    left, top, scale_x, scale_y = transform
    height, width = frame_shape[:2]
    corners = boxes[:, selected].T.astype(np.float32)
    corners -= (left, top, left, top)
    corners /= (scale_x, scale_y, scale_x, scale_y)
    corners = np.clip(corners, 0, (width - 1, height - 1, width - 1, height - 1))
    return corners.round().astype(int), class_ids[selected], best_scores[selected]


class YOLODetector:
    def __init__(self):
        model = MODEL_PATH
        if not model.is_file():
            raise RuntimeError(f"Missing YOLO model: {model}")

        from rknnlite.api import RKNNLite

        self.runtime = RKNNLite()
        try:
            if self.runtime.load_rknn(str(model)) != 0:
                raise RuntimeError(f"Could not load YOLO model: {model}")
            if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                raise RuntimeError("Could not initialize the RK3588 NPU for YOLO")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.runtime.release()

    def __call__(self, frame, preview=False):
        tensor, transform = prepare(frame)
        outputs = self.runtime.inference(inputs=[tensor], data_format=["nhwc"])
        boxes, classes, scores = decode(outputs, 0.25, transform, frame.shape)
        labels = []
        for (x1, y1, x2, y2), class_id, score in zip(boxes, classes, scores):
            if x2 <= x1 or y2 <= y1:
                continue
            label = f"{CLASSES[class_id]} {score:.2f}"
            labels.append(f"{label} ({x1}, {y1}, {x2}, {y2})")
            if preview:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return "\n".join(labels)
