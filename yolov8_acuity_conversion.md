# YOLOv8 → NPU (.nb) Conversion via ACUITY Toolkit

> Agent-ready steps. Run on **x86 Linux host** inside the Allwinner Docker container.

---

## Prerequisites (Host)

```bash
# Export YOLOv8 ONNX with ultralytics (on host, NOT in docker)
pip install ultralytics onnxsim

python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26s.pt')          # your weights file
model.export(format='onnx', imgsz=640, opset=12, simplify=False, dynamic=False)
"
# This produces yolo26s.onnx

# Simplify (fixes static input shape for NPU)
onnxsim yolo26s.onnx yolo26s-sim.onnx --input-shape 1,3,640,640
```

---

## Phase 1: Prepare workspace

```bash
# Inside your ai-sdk docker_data directory (mapped into container)
mkdir -p docker_data/ai-sdk/models/yolo26s-sim/images

# Copy files in
cp yolo26s-sim.onnx docker_data/ai-sdk/models/yolo26s-sim/
cp <your_calibration_images/*.jpg> docker_data/ai-sdk/models/yolo26s-sim/images/
```

---

## Phase 2: Create config files

### `docker_data/ai-sdk/models/yolo26s-sim/dataset.txt`
List 10–20 representative calibration images (paths relative to models/ dir):
```
./yolo26s-sim/images/img1.jpg
./yolo26s-sim/images/img2.jpg
./yolo26s-sim/images/img3.jpg
...
```

### `docker_data/ai-sdk/models/yolo26s-sim/inputs_outputs.txt`

> ⚠️ **This is the critical difference from YOLOv5!**
> YOLOv8 has a SINGLE output head named `output0` (not 3 numbered heads like `350 498 646`)

```
--inputs images --input-size-list '3,640,640' --outputs 'output0'
```

### `docker_data/ai-sdk/models/yolo26s-sim/yolo26s-sim_inputmeta.yml`

> ⚠️ Check the actual `lid` name after running `pegasus_import.sh` — it may be `images_xxx` with a number suffix.

```yaml
input_meta:
  databases:
  - path: dataset.txt
    type: TEXT
    ports:
    - lid: images        # ← change to images_394 etc if import fails
      category: image
      dtype: float32
      sparse: false
      layout: nchw
      shape:
      - 1
      - 3
      - 640
      - 640
      fitting: scale
      preprocess:
        reverse_channel: true    # BGR → RGB
        mean:
        - 0
        - 0
        - 0
        scale:
        - 0.00392157             # 1/255
        - 0.00392157
        - 0.00392157
        preproc_node_params:
          add_preproc_node: false
          preproc_type: IMAGE_RGB
          preproc_image_size:
          - 640
          - 640
          preproc_crop:
            enable_preproc_crop: false
            crop_rect:
            - 0
            - 0
            - 640
            - 640
          preproc_perm:
          - 0
          - 1
          - 2
          - 3
      redirect_to_output: false
```

---

## Phase 3: Run Conversion (inside Docker)

```bash
sudo docker exec -it -w /workspace/ai-sdk/models allwinner_v2.0.10 /bin/bash -c "
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3      # v3 = A733, v2 = T527

cp ../scripts/* .

# Step 1: Import ONNX → IR
./pegasus_import.sh yolo26s-sim/

# ⚠️ After import, check the actual input layer name:
# grep -A2 '\"op\": \"input\"' yolo26s-sim/yolo26s-sim.json | head -10
# Update lid in yolo26s-sim_inputmeta.yml if it's 'images_xxx'

# Step 2: Quantize to uint8
./pegasus_quantize.sh yolo26s-sim/ uint8 10

# Step 3: Export to .nb
./pegasus_export_ovx.sh yolo26s-sim/ uint8
"
```

---

## Phase 4: Get the .nb file

```bash
# The output is always at this path:
ls docker_data/ai-sdk/models/yolo26s-sim/wksp/yolo26s-sim_uint8_nbg_unify/network_binary.nb

# Copy to your project
cp docker_data/ai-sdk/models/yolo26s-sim/wksp/yolo26s-sim_uint8_nbg_unify/network_binary.nb \
   ~/Project/main-project-sw/models/yolo26s_new.nb
```

---

## Troubleshooting

### `lid` mismatch in quantize step
```bash
# Run this after pegasus_import.sh to find the real input name
grep -A2 '"op": "input"' yolo26s-sim/yolo26s-sim.json | head -20
# You'll see something like: "images_394"
# Update yolo26s-sim_inputmeta.yml: lid: images_394
```

### Verifying output node name
```bash
# Use netron or python to check output names:
python3 -c "
import onnx
m = onnx.load('yolo26s-sim.onnx')
for o in m.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
"
# For standard YOLOv8: output0   [1, 84, 8400]
# For YOLOv8-seg:      output0   [1, 116, 8400]
#                      output1   [1, 32, 160, 160]
```

### For the segmentation model (`yolo26_seg.nb`)
The `inputs_outputs.txt` needs BOTH outputs:
```
--inputs images --input-size-list '3,640,640' --outputs 'output0 output1'
```

---

## What the existing `yolo26s.nb` output looks like

Based on the model dimensions: the NPU exports output as `[84, 8400]` (84 channels × 8400 detections).  
Running the updated `yolov8` binary will now print diagnostic info showing the exact tensor values — this will confirm whether the model was exported correctly.
