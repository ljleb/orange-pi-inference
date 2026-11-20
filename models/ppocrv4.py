"""Run Rockchip's PP-OCRv4 detection and recognition models through RKNN Lite2."""
from pathlib import Path
import sys

import cv2
import numpy as np


class RKNNTextReader:
    def __init__(self, *, crop_only=False):
        local = Path(__file__).resolve().parents[1] / "local"
        vendor = local / "vendor" / "rknn_model_zoo" / "examples" / "PPOCR" / "PPOCR-System"
        if not (vendor / "python" / "ppocr_system.py").is_file():
            raise RuntimeError(f"Missing Rockchip PPOCR example at {vendor}")
        sys.path.insert(0, str(vendor / "python"))
        try:
            from ppocr_system import get_rotate_crop_image, sorted_boxes
            from utils.db_postprocess import DBPostProcess, DetPostProcess
            from utils.rec_postprocess import CTCLabelDecode
        finally:
            sys.path.pop(0)
        self.crop = get_rotate_crop_image
        self.sort_boxes = sorted_boxes
        self.detect_boxes = DBPostProcess(thresh=.3, box_thresh=.6, max_candidates=1000,
                                         unclip_ratio=1.5, use_dilation=False, score_mode="fast")
        self.filter_boxes = DetPostProcess()
        self.decode = CTCLabelDecode(
            character_dict_path=str(vendor / "model" / "ppocr_keys_v1.txt"), use_space_char=True
        )

        from rknnlite.api import RKNNLite

        self.detector = None if crop_only else RKNNLite()
        self.recognizer = RKNNLite()
        try:
            for runtime, name in ((self.detector, "det"), (self.recognizer, "rec")):
                if runtime is None:
                    continue
                model = local / "models" / "ppocrv4" / f"ppocrv4_{name}.rknn"
                if not model.is_file() or runtime.load_rknn(str(model)) != 0:
                    raise RuntimeError(f"Could not load OCR model {model}")
                if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                    raise RuntimeError(f"Could not initialize the NPU for {model}")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.recognizer.release()
        if self.detector is not None:
            self.detector.release()

    @staticmethod
    def infer(runtime, image):
        outputs = runtime.inference(inputs=[image[None]], data_format=["nhwc"])
        if not outputs:
            raise RuntimeError("PP-OCRv4 RKNN inference failed")
        return outputs[0].astype(np.float32)

    def __call__(self, frame):
        if self.detector is None:
            raise RuntimeError("Crop-only OCR requires recognize_crop()")
        if frame is None:
            raise RuntimeError("Could not read OCR image")
        height, width = frame.shape[:2]
        scale = min(1.0, 480 / max(height, width))
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        padded = np.zeros((480, 480, 3), dtype=np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
        maps = self.infer(self.detector, padded)
        # Decode into detector pixels, clip the padded area, then map to the source.
        shape = np.array([[480, 480, 1, 1]], dtype=np.float32)
        points = self.detect_boxes({"maps": maps}, shape)[0]["points"]
        if not len(points):
            return ""
        points = points.astype(np.float32)
        points *= (width / resized_width, height / resized_height)
        points = self.filter_boxes.filter_tag_det_res(points, frame.shape)

        lines = []
        for box in self.sort_boxes(points):
            crop = self.crop(frame, box.astype(np.float32))
            text, confidence = self.recognize_crop(crop)
            if text and confidence >= 0.5:
                lines.append(text)
        return "\n".join(lines)

    def recognize_crop(self, crop):
        normalized = cv2.resize(crop, (320, 48)).astype(np.float32) / 255.0
        output = self.infer(self.recognizer, normalized)
        return self.decode(output)[0]
