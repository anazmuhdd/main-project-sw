"""
YOLOv5 Webcam Object Detection (Pretrained COCO)
=================================================
Standalone script for live webcam inference using the standard pretrained
YOLOv5s model from Ultralytics (80 COCO classes, detection only — no masks).

Pipeline (mirrors yolov5/detect.py exactly):
  1.  letterbox preprocessing (aspect-ratio-preserving resize + pad)
  2.  model forward  →  detection predictions
  3.  NMS (nm=0, standard detection)
  4.  scale_boxes  →  map back to original frame
  5.  annotator.box_label()  →  bounding boxes + class names
"""

import os
import sys
import cv2
import numpy as np
import torch
import time
from pathlib import Path

# ── Backend setup ──────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
BACKEND_ROOT = FILE.parents[0]  # backend/yolov5_object_detection/
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors


# ───────────────────────────────────────────────────────────────────────────
def preprocess(frame, imgsz, stride, pt, device, half):
    """Exact YOLOv5 preprocessing: letterbox → HWC→CHW → BGR→RGB → tensor."""
    img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]        # HWC → CHW, BGR → RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


# ───────────────────────────────────────────────────────────────────────────
def run_detection(
    weights="yolov5s.pt",
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="0",
):
    """
    Main webcam detection loop using the pretrained YOLOv5s model.
    The weights file will be auto-downloaded by ultralytics if not present.
    """
    # ── 1. Device ──────────────────────────────────────────────────────────
    device = select_device(device)
    half = device.type != "cpu"

    # ── 2. Model ───────────────────────────────────────────────────────────
    print(f"Loading model: {weights}")
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=half)
    stride = int(model.stride)
    names = model.names               # {0:'person', 1:'bicycle', ...} — 80 COCO classes
    pt = model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Warmup
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    print(f"Model loaded — {len(names)} classes")

    # ── 3. Webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    print("Live feed active.  Press 'q' to stop.")

    # ── 4. Inference loop ──────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # ── preprocess ────────────────────────────────────────────────────
        img = preprocess(frame, imgsz, stride, pt, device, half)

        # ── inference ─────────────────────────────────────────────────────
        pred = model(img, augment=False, visualize=False)

        # ── NMS (standard detection, no masks → nm=0) ─────────────────────
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000,
        )

        t1 = time.time()
        fps = 1.0 / max(t1 - t0, 1e-9)

        # ── render ────────────────────────────────────────────────────────
        for det in pred:
            annotator = Annotator(frame, line_width=2, example=str(names))

            if len(det):
                # Rescale boxes from model input size → original frame
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], frame.shape
                ).round()

                # Draw bounding boxes + labels
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    name = (
                        names.get(c, f"class_{c}")
                        if isinstance(names, dict)
                        else names[c]
                    )
                    label = f"{name} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            rendered = annotator.result()
            cv2.putText(
                rendered,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YOLOv5 Object Detection (COCO)", rendered)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping…")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
