# RoboMaster Armor Detection / Keypoints (YOLOv7-style)

## About This Repository

This repository is for **personal learning and practice** only—to get familiar with RoboMaster-style vision workflows (detection, export, simple deployment demos, etc.). It is **not** affiliated with any official team or competition.

---

This project is based on YOLOv7-style code and supports **armor plate object detection** (e.g. 12 classes), **four-corner armor keypoints**, energy-target-style configs, and more. Unless stated otherwise, commands are run from the **repository root** (after cloning, `cd` into that directory).

---

## Repository Layout (common paths)

| Path | Description |
|------|-------------|
| `train.py` | Training entry point |
| `detect.py` | Inference on images / video / webcam |
| `export.py` | Export ONNX / TorchScript, etc. |
| `demo_webcam.py` | Live webcam: detection + PnP + Kalman (optional JSON/UDP) |
| `dataset/data.yaml` | Dataset config (paths, `nc`, `names`) |
| `dataset/train/images`, `dataset/train/labels` | Training images and YOLO labels |
| `dataset/valid/`, `dataset/test/` | Validation / test splits |
| `cfg/armor_detect_12.yaml` | 12-class detection-only model |
| `cfg/armor_keypoints_14.yaml` | 14-class four-keypoint model |
| `hyp/hyp.scratch.p5.yaml`, etc. | Hyperparameters |
| `runs/train/<run_name>/` | Training outputs (`weights/best.pt`, `results.txt`, etc.) |
| `C++_inference_openvino_kpt/` | OpenVINO C++ demo (configure model path, class count, etc.) |

---

## 1. Environment

1. Install **Python 3.8+** (match PyTorch’s requirements).
2. Install **PyTorch** (CUDA build must match your GPU driver).
3. Enter the project directory and install dependencies (use `requirements.txt` if present; otherwise install at least `opencv-python`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`, and anything else missing when you run training).

```powershell
cd <path-to-your-repo-root>
conda activate your_env   # e.g. yolov7
```

---

## 2. Dataset

1. **YOLO format**: each image has a matching `labels/*.txt`; each line is `class x y w h` (normalized center and size).
2. Configure `dataset/data.yaml`:
   - `train` / `val`: image directories (paths relative to repo root)
   - `nc`: number of classes
   - `names`: class names in the same order as label IDs
3. If class count or order differs from the bundled examples, update `nc` in the matching `cfg/*.yaml` or keep it consistent with `data.yaml`.

Keypoint dataset layout follows the original upstream README (armor four corners need extra labels).

---

## 3. Training

### 3.1 Single-GPU (12-class armor detection example)

Default `--cfg` / `--data` / `--hyp` in `train.py` may point to paths that do not exist in your tree—**pass explicit paths on the command line**.

```powershell
python train.py --workers 8 --device 0 --batch-size 32 --data dataset/data.yaml --img-size 640 640 --cfg cfg/armor_detect_12.yaml --name armor12 --hyp hyp/hyp.scratch.p5.yaml --project runs/train
```

Notes:

- **`--weights`**: omit to train from scratch; for pretrained weights use e.g. `--weights yolov7.pt` (path on your machine).
- **`--img-size`**: should match inference/export as closely as possible.
- **`--project`**: using `runs/train` avoids writing outside the repo by default.
- **`--exist-ok`**: add if you want to reuse the same run name without auto-incrementing the folder name.
- On Windows, YAML files with Chinese comments are opened with **`encoding='utf-8'`** in relevant scripts; if issues persist, set `PYTHONUTF8=1`.

### 3.2 Multi-GPU (common on Linux)

```bash
# Example: 4 GPUs; total batch size must be divisible by GPU count
export USE_LIBUV=0   # On Windows, try this if distributed init fails with libuv
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data dataset/data.yaml --img-size 640 640 --cfg cfg/armor_detect_12.yaml --name armor12 --hyp hyp/hyp.scratch.p5.yaml --project runs/train
```

### 3.3 Resume training

```powershell
python train.py --resume runs/train/armor12/weights/last.pt
```

(Exact behavior follows `train.py`’s `--resume` implementation.)

### 3.4 Training outputs

- **Weights**: `runs/train/<name>/weights/best.pt`, `last.pt`
- **Curves / metrics**: `results.txt`, `*.png`, etc. in the same run folder

---

## 4. Inference

### 4.1 Images / video / webcam

Use `detect.py` with `--weights` set. Match the model type with **`--kpt-label`** (`False` for detection-only, `True` for keypoint models).

```powershell
# Image or directory
python detect.py --weights runs/train/armor12/weights/best.pt --source path/to/img.jpg --img-size 640 --device 0 --view-img --kpt-label False

# Webcam (device index 0)
python detect.py --weights runs/train/armor12/weights/best.pt --source 0 --img-size 640 --device 0 --view-img --kpt-label False
```

`--project` and `--name` control where results are saved.

### 4.2 Live webcam demo (bbox + optional PnP / tracking)

For **detection-only** weights, `demo_webcam.py` can use **bounding-box corners** as a rough stand-in for PnP (demo only—not real light-bar corners). **Keypoint** models use network-predicted corners.

```powershell
python demo_webcam.py --weights runs/train/armor12/weights/best.pt --img-size 640 --device 0
```

Useful flags:

- `--calib`: camera calibration YAML (`camera_matrix`, `dist_coeffs`)
- `--fov-deg` / `--fx`: approximate intrinsics without a calib file
- `--max-missed`: how many frames to keep a Kalman-predicted box after a missed detection
- `--stream-json` / `--udp`: optional JSON telemetry to stdout or UDP

Press **`q`** to quit the window.

---

## 5. Export ONNX (deployment)

`export.py` converts `best.pt` to **ONNX** for OpenVINO, ONNX Runtime, TensorRT, etc.

From the repository root:

```powershell
python export.py --weights runs/train/armor12/weights/best.pt --img-size 640 640 --device 0
```

Notes:

- On success, **`best.onnx`** is written next to the weights (see `export.py` for details).
- Requires **`onnx`**; optional simplify needs **`onnx-simplifier`**.
- **`--img-size`** should match training / deployment.
- For keypoint exports with `--export-nms` and similar, read `export.py` and keep `kpt_label` consistent.

---

## 6. C++ OpenVINO (optional)

`C++_inference_openvino_kpt/` is an **Intel OpenVINO** sample: convert ONNX to IR (`.xml` + `.bin`), then set **`MODEL_PATH`**, **`IMG_SIZE`**, **`CLS_NUM`**, **`KPT_NUM`** (0 for detection-only), **anchors**, and OpenCV/OpenVINO paths in `yolov7_kpt.h` / `CMakeLists.txt`.

For detection-only models with no keypoints, you can fill **`Object::kpt`** from **bbox corners** in C++ for downstream PnP (same idea as the Python demo).

---

## 7. Suggested workflow: train → vehicle

1. Prepare data; keep `dataset/data.yaml` and `cfg` **`nc`** aligned.  
2. Run `train.py` to obtain `best.pt`.  
3. Validate with `detect.py` or `demo_webcam.py` in real imaging conditions.  
4. Run `export.py` to get `best.onnx`.  
5. Deploy with **ONNX Runtime / OpenVINO / TensorRT**, etc., and wire outputs to your robot middleware.  
6. For ranging / PnP, **calibrate the camera** before competition and load `camera_matrix` and distortion in code.

---

## 8. Troubleshooting

| Symptom | What to try |
|---------|----------------|
| `--weights ""` fails in PowerShell | Omit `--weights`, or use `--weights=''` |
| Windows multi-GPU `libuv` error | Run `$env:USE_LIBUV="0"` before launching distributed training |
| YAML `UnicodeDecodeError` | Save YAML as UTF-8; this repo opens many files with `encoding='utf-8'` |
| Folders like `armor12`, `armor122` under `runs/train` | Auto-increment when the run name exists; add `--exist-ok` to overwrite |
| Bounding box flicker | Use tracking/Kalman in `demo_webcam.py`; tune confidence; improve lighting / focus |

---

## 9. References and licensing

- Upstream framework and dataset notes may appear in the original author’s documentation and RoboMaster community guidelines.  
- If data comes from Roboflow or similar sources, follow their license (see comments in `dataset/data.yaml`).

For issues, cross-check this README with `train.py`, `detect.py`, `export.py`, and `demo_webcam.py` (`--help`).
