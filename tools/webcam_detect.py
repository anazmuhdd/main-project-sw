import os
import sys
import cv2
import torch
import time
from pathlib import Path

# Add yolov5 directory to sys.path for internal imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

def run_webcam(weights='models/yolov5_currency.pt', imgsz=640, conf_thres=0.5, iou_thres=0.45, device='0'):
    # Select Device
    device = select_device(device)
    
    # Load Model
    print(f"Loading model {weights}...")
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Note: Using robust name lookup (handling missing indices)
    # names = {0: 'n10', 1: 'n100', ...}

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Live Detection Started. Press 'q' to quit. ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time.time()
        pred = model(img, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic_nms=False, max_det=1000)
        t2 = time.time()
        
        fps = 1.0 / (t2 - t1)

        # Process predictions
        for i, det in enumerate(pred):  
            annotator = Annotator(frame, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # Robust name lookup
                    name = names.get(c, f"class_{c}")
                    label = f"{name} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Display resulting frame
            img0 = annotator.result()
            cv2.putText(img0, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv5 Currency Detection", img0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Custom weights and device
    run_webcam(weights='models/yolov5_currency.pt', device='0')
