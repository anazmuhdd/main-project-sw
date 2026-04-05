"""
YOLOv5 Webcam Segmentation Inference
=====================================
Standalone script for live webcam inference using a YOLOv5 *segmentation*
model.  Follows the exact pipeline from  yolov5/segment/predict.py:

  1.  letterbox preprocessing (aspect-ratio-preserving resize + pad)
  2.  model forward  →  (predictions, prototype_masks)
  3.  NMS with nm=32 (32 mask coefficient columns)
  4.  process_mask  →  binary masks upsampled to input size
  5.  annotator.masks()  →  coloured overlay on the frame
  6.  annotator.box_label()  →  bounding boxes + class names
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
BACKEND_ROOT = FILE.parents[0]  # backend/yolo_v5_running_tools/
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
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
    weights="../../models/yolov5_currency.pt",
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="0",
):
    # ── 1. Device ──────────────────────────────────────────────────────────
    device = select_device(device)
    half = device.type != "cpu"

    # ── 2. Model ───────────────────────────────────────────────────────────
    weights_path = str((BACKEND_ROOT / weights).resolve())
    # print(f"Loading model: {weights_path}")
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=half)
    stride = int(model.stride)
    names = model.names                          # {0:'n10', 1:'n100', …}
    pt = model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Warmup
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    # print(f"Model loaded — classes: {names}")
    # print(f"Segmentation model detected (nm=32).")

    # ── 3. Webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    # print("Live feed active.  Press 'q' to stop.")

    # ── 4. Inference loop ──────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # ── preprocess ────────────────────────────────────────────────────
        img = preprocess(frame, imgsz, stride, pt, device, half)

        # ── inference ─────────────────────────────────────────────────────
        # Segmentation models return (pred, proto_masks, ...)
        out = model(img, augment=False, visualize=False)
        pred = out[0]        # (1, N, 5+nc+nm)  — box + obj + cls + masks
        proto = out[1]       # (1, 32, 160, 160) — prototype masks

        # ── NMS ───────────────────────────────────────────────────────────
        # nm=32 tells NMS to keep the 32 mask-coefficient columns
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000,
            nm=32,
        )

        t1 = time.time()
        fps = 1.0 / max(t1 - t0, 1e-9)

        # ── render ────────────────────────────────────────────────────────
        for i, det in enumerate(pred):
            annotator = Annotator(frame, line_width=2, example=str(names))

            if len(det):
                # Process masks BEFORE rescaling boxes (as in predict.py)
                masks = process_mask(
                    proto[i],          # (32, 160, 160)
                    det[:, 6:],        # mask coefficients per detection
                    det[:, :4],        # boxes in model-input coords
                    img.shape[2:],     # model input size (H, W)
                    upsample=True,
                )

                # Rescale boxes from model-input size → original frame
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], frame.shape
                ).round()

                # Draw coloured segmentation masks
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=img[i],     # normalised CHW tensor on GPU
                )

                # Draw bounding boxes + labels on top
                for *xyxy, conf, cls in reversed(det[:, :6]):
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
            cv2.imshow("YOLOv5 Currency Segmentation", rendered)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping…")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
