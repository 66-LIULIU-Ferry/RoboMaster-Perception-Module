# RoboMaster 装甲板检测 / 关键点（YOLOv7 系）使用说明

本仓库基于 YOLOv7 系代码，支持 **装甲板目标检测**（如 12 类）、**装甲板四点关键点**、能量机关等配置。以下步骤默认在 **仓库根目录** 下执行（克隆本仓库后 `cd` 到该目录即可）。

---

## 目录结构（常用）

| 路径 | 说明 |
|------|------|
| `train.py` | 训练入口 |
| `detect.py` | 图片 / 视频 / 摄像头推理 |
| `export.py` | 导出 ONNX / TorchScript 等 |
| `demo_webcam.py` | 摄像头实时检测 + PnP + 卡尔曼（可选 JSON/UDP） |
| `dataset/data.yaml` | 数据集配置（路径、`nc`、`names`） |
| `dataset/train/images`、`dataset/train/labels` | 训练集图片与 YOLO 标签 |
| `dataset/valid/`、`dataset/test/` | 验证集 / 测试集 |
| `cfg/armor_detect_12.yaml` | 12 类纯检测模型结构 |
| `cfg/armor_keypoints_14.yaml` | 14 类四点关键点模型结构 |
| `hyp/hyp.scratch.p5.yaml` 等 | 超参数 |
| `runs/train/<实验名>/` | 训练输出（`weights/best.pt`、`results.txt` 等） |
| `C++_inference_openvino_kpt/` | OpenVINO C++ 推理示例（需自行配置模型路径与类别数） |

---

## 一、环境准备

1. 安装 **Python 3.8+**（与 PyTorch 官方说明一致即可）。
2. 安装 **PyTorch**（CUDA 版本需与本机显卡驱动匹配）。
3. 进入项目目录，安装依赖（若仓库有 `requirements.txt` 则按文件安装；否则至少包含 `opencv-python`、`numpy`、`pyyaml`、`tqdm`、`matplotlib` 等，与训练报错提示补齐）。

```powershell
cd <你的仓库根目录>
conda activate your_env   # 例如 yolov7
```

---

## 二、数据集准备

1. **YOLO 格式**：每张图片对应同名 `labels/*.txt`，每行：`class x y w h`（归一化中心坐标与宽高）。
2. 在 `dataset/data.yaml` 中配置：
   - `train` / `val`：指向图片目录（相对仓库根目录）
   - `nc`：类别数
   - `names`：类别名列表，顺序与标签 id 一致
3. 若类别数或顺序与官方示例不同，请同步修改 `cfg/` 中对应 yaml 的 `nc`，或保持与 `data.yaml` 一致。

关键点数据集格式见原仓库说明（装甲四点等需额外标注）。

---

## 三、训练

### 3.1 单卡训练（12 类装甲检测示例）

`train.py` 中默认 `--cfg` / `--data` / `--hyp` 可能指向不存在的路径，**建议在命令行中显式写出本仓库实际路径**。

```powershell
python train.py --workers 8 --device 0 --batch-size 32 --data dataset/data.yaml --img-size 640 640 --cfg cfg/armor_detect_12.yaml --name armor12 --hyp hyp/hyp.scratch.p5.yaml --project runs/train
```

说明：

- **`--weights`**：不写表示从头训练；若需预训练权重，使用 `--weights yolov7.pt` 等（路径按你本地文件）。
- **`--img-size`**：训练输入尺寸，需与后续推理、导出尽量一致。
- **`--project`**：建议写 `runs/train`，避免默认路径指向仓库外。
- **`--exist-ok`**：需要覆盖同一实验名时可加。
- Windows 下若读 yaml 含中文注释，已在本仓库相关脚本中使用 **`encoding='utf-8'`** 打开；若仍报错，可设置环境变量 `PYTHONUTF8=1`。

### 3.2 多卡分布式（Linux 常见）

```bash
# 示例：4 卡；总 batch 需能被 GPU 数整除
export USE_LIBUV=0   # Windows 下若遇到 libuv 报错可尝试
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data dataset/data.yaml --img-size 640 640 --cfg cfg/armor_detect_12.yaml --name armor12 --hyp hyp/hyp.scratch.p5.yaml --project runs/train
```

### 3.3 断点续训

```powershell
python train.py --resume runs/train/armor12/weights/last.pt
```

（具体以 `train.py` 中 `--resume` 逻辑为准。）

### 3.4 训练产出

- **权重**：`runs/train/<name>/weights/best.pt`、`last.pt`
- **曲线与指标**：同目录下 `results.txt`、`*.png` 等

---

## 四、推理测试

### 4.1 图片 / 视频 / 摄像头

使用 `detect.py`，需指定 `--weights`，并与模型类型一致设置 **`--kpt-label`**（纯检测为 `False`，关键点为 `True`）。

```powershell
# 图片或目录
python detect.py --weights runs/train/armor12/weights/best.pt --source path/to/img.jpg --img-size 640 --device 0 --view-img --kpt-label False

# 摄像头（source 为 0）
python detect.py --weights runs/train/armor12/weights/best.pt --source 0 --img-size 640 --device 0 --view-img --kpt-label False
```

`--project`、`--name` 控制保存目录。

### 4.2 摄像头实时演示（检测框 + 可选 PnP / 跟踪）

`demo_webcam.py` 在纯检测模型下可用 **检测框四角** 近似参与 PnP（仅演示，非真实灯条角点）；关键点模型则使用网络输出的角点。

```powershell
python demo_webcam.py --weights runs/train/armor12/weights/best.pt --img-size 640 --device 0
```

常用参数：

- `--calib`：相机标定 yaml（`camera_matrix`、`dist_coeffs`）
- `--fov-deg` / `--fx`：无标定时的近似内参
- `--max-missed`：检测丢失后卡尔曼预测框保持的帧数
- `--stream-json` / `--udp`：向 stdout 或 UDP 输出 JSON（可选）

按 **`q`** 退出窗口。

---

## 五、导出 ONNX（部署用）

训练得到的 `best.pt` 可通过 `export.py` 导出为 **ONNX**，便于 OpenVINO、ONNX Runtime、TensorRT 等使用。

在仓库根目录执行：

```powershell
python export.py --weights runs/train/armor12/weights/best.pt --img-size 640 640 --device 0
```

说明：

- 导出成功后在权重同目录生成 **`best.onnx`**（与 `export.py` 内逻辑一致）。
- 需安装 **`onnx`**；若开启 simplify，需 **`onnx-simplifier`**。
- **`--img-size`** 应与训练 / 部署输入一致。
- 关键点模型导出时若使用 `--export-nms` 等选项，需阅读 `export.py` 内说明并与 `kpt_label` 一致。

---

## 六、C++ OpenVINO 推理（可选）

目录 `C++_inference_openvino_kpt/` 为 **Intel OpenVINO** 示例：需将 ONNX 转为 OpenVINO IR（`.xml` + `.bin`），并在 `yolov7_kpt.h` 中配置 **`MODEL_PATH`**、**`IMG_SIZE`**、**`CLS_NUM`**、**`KPT_NUM`**（纯检测为 0）及 **anchor** 与 `CMakeLists.txt` 中 OpenCV/OpenVINO 路径。

纯检测模型无关键点时，可在 C++ 侧用 **检测框四角** 填充 `Object::kpt` 供后续 PnP（与 Python 思路一致）。

---

## 七、从训练到上车的推荐流程

1. 准备数据与 `dataset/data.yaml`、`cfg` 中 `nc` 一致。  
2. 运行 `train.py`，得到 `best.pt`。  
3. 用 `detect.py` 或 `demo_webcam.py` 在实拍环境验证。  
4. 运行 `export.py` 得到 `best.onnx`。  
5. 按车载方案选择 **ONNX Runtime / OpenVINO / TensorRT** 等，将后处理与电控通信对接。  
6. 需要测距 / PnP 时，赛前完成 **相机标定**，在代码中加载 `camera_matrix` 与畸变系数。

---

## 八、常见问题

| 现象 | 处理建议 |
|------|----------|
| PowerShell 中 `--weights ""` 报错 | 去掉 `--weights`，或写成 `--weights ''`（等号形式） |
| Windows 多卡 `libuv` 报错 | 先执行 `$env:USE_LIBUV="0"` 再启动 distributed |
| 训练时 yaml 解码错误 | 保证 yaml 为 UTF-8；本仓库已对多处 `open` 使用 `encoding='utf-8'` |
| `runs/train` 下出现 `armor12`、`armor122` 多个目录 | 同名实验未加 `--exist-ok` 时自动递增；覆盖可加 `--exist-ok` |
| 检测框闪烁 | 使用 `demo_webcam.py` 中跟踪与卡尔曼；或提高置信度、改善光照/对焦 |

---

## 九、参考与许可

- 原始框架与数据集说明可参考仓库内原始作者说明及 RoboMaster 社区规范。  
- 数据集若来自 Roboflow 等，请遵守对应许可证（见 `dataset/data.yaml` 中注释）。

如有问题，请结合本 README 与 `train.py`、`detect.py`、`export.py`、`demo_webcam.py` 内 `--help` 参数逐项核对。
