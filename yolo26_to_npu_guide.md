# 📖 YOLO26 to NPU (.nb) Conversion Guide

This guide provides the complete set of commands and steps required to convert a trained YOLO26 (Ultralytics family) model to the NPU-optimized `.nb` format for the Radxa Cubie A7Z (Allwinner A733 chip).

---

## 🛠️ Phase 1: Export and Simplify (Host Machine)

Perform these steps on your host machine where you have `ultralytics` and `onnx-simplifier` installed.

### 1. Export to ONNX

Replace `your_model.pt` with your trained weights file.

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('your_model.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=False, dynamic=False)
"
```

### 2. Simplify Model

This fixes the input dimensions and prepares the graph for the ACUITY Toolkit.

```bash
pip3 install onnxsim
onnxsim your_model.onnx your_model-sim.onnx --input-shape 1,3,640,640
```

---

## 📁 Phase 2: Workspace Setup (Host Machine)

Create a dedicated directory for the model conversion inside your mapped docker data folder.

### 1. Create Directories

```bash
mkdir -p docker_data/ai-sdk/models/your_model-sim
```

### 2. Create Configuration Files

Move the simplified ONNX model into this directory and create these files:

#### `inputs_outputs.txt`

```text
images
output0
```

#### `dataset.txt`

This file should list the paths to 10-20 representative images for quantization. You can reuse the existing COCO images:

```text
./images/000000000089.jpg
./images/000000000139.jpg
./images/000000000285.jpg
./images/000000000632.jpg
./images/000000000724.jpg
...
```

#### `your_model-sim_inputmeta.yml`

Update the `lid` based on the imported layer name (see troubleshooting below).

```yaml
# input information
- lid: images
  type: tensor
  dataset:
    - path: dataset.txt
      type: TEXT
      ports:
        - lid: images
          category: image
          dtype: float32
          sparse: false
          shape: [1, 3, 640, 640]
          mean: [0, 0, 0]
          scale: [0.0039215686, 0.0039215686, 0.0039215686] # 1/255
          channel_reverse: true # BGR to RGB
```

---

## 🚀 Phase 3: NPU Compilation (Docker Container)

Run this final combined command from your project root. It handles environment setup, import, quantization, and export.

```bash
sudo docker exec -it -w /workspace/ai-sdk/models allwinner_v2.0.10 /bin/bash -c "
export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
source env.sh v3
cp ../scripts/* .

# 1. Import
./pegasus_import.sh your_model-sim/

# 2. Quantize (Edit inputmeta.yml if lid mismatch occurs)
./pegasus_quantize.sh your_model-sim/ uint8 10

# 3. Export to NB
./pegasus_export_ovx.sh your_model-sim/ uint8
"
```

---

## ⚠️ Troubleshooting & Lessons Learned

### 1. Input Layer Name Mismatch (`lid`)

If `pegasus_import.sh` finishes but `pegasus_quantize.sh` fails with a "tensor not found" error, check the layer name in the imported model:

- Look at the `pegasus_import.sh` output log for: `Process images_### ...`
- Open `docker_data/ai-sdk/models/your_model-sim/your_model-sim.json` and search for `"op": "input"`.
- Update the `lid` in your `your_model-sim_inputmeta.yml` to match (e.g., `images_394`).

### 2. User Permissions

Always run the `docker exec` command with `sudo` if your user is not in the `docker` group.

### 3. Allwinner Chip Versions

- Use `source env.sh v3` for Radxa Cubie A7Z (Allwinner A733).
- Use `source env.sh v2` for T527 chips.

### 4. Output Format

The final model will always be named `network_binary.nb` and located inside:
`your_model-sim/wksp/your_model-sim_uint8_nbg_unify/`
