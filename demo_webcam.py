"""
摄像头装甲板演示：2D 点 -> PnP 位姿 -> 卡尔曼平滑
"""
import argparse
import json
import socket
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized


# 小装甲板四角在「装甲坐标系」下的 3D 点（单位：米），顺序与标注一致：左上、左下、右下、右上
# 以下为约 125mm x 55mm 板面、中心在原点、法向为 +Z 的平面模型，务必按实测修改。
_HALF_W, _HALF_H = 0.0625, 0.0275
SMALL_ARMOR_POINTS_3D = np.array(
    [
        [-_HALF_W, -_HALF_H, 0.0],
        [-_HALF_W, _HALF_H, 0.0],
        [_HALF_W, _HALF_H, 0.0],
        [_HALF_W, -_HALF_H, 0.0],
    ],
    dtype=np.float64,
)


def build_camera_matrix(w: int, h: int, fov_deg: float = 55.0) -> np.ndarray:
    """无标定、无 fx 时：用水平视场角粗估 fx=fy（仅演示）。"""
    fx = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_camera_matrix_intrinsics(
    w: int,
    h: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> np.ndarray:
    """用 fx、fy（像素）与主点 cx、cy 组装 K；未给 fy 则 fy=fx。"""
    if fx is None:
        raise ValueError("fx 不能为空")
    fxf = float(fx)
    fyf = float(fy) if fy is not None else fxf
    cxf = float(cx) if cx is not None else w / 2.0
    cyf = float(cy) if cy is not None else h / 2.0
    return np.array([[fxf, 0.0, cxf], [0.0, fyf, cyf], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_calib_yaml(path: str) -> tuple:
    """从 yaml 读取 camera_matrix（3x3）与 dist_coeffs（k1,k2,p1,p2,k3 等）。"""
    import yaml

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    k = np.array(d["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.array(d.get("dist_coeffs", [[0, 0, 0, 0, 0]]), dtype=np.float64).ravel()
    return k, dist


def bbox_to_img_points_four_corners(xyxy: np.ndarray) -> np.ndarray:
    """轴对齐框四角，顺序与 SMALL_ARMOR_POINTS_3D 一致：左上、左下、右下、右上。"""
    x1, y1, x2, y2 = xyxy.tolist()
    return np.array(
        [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
        dtype=np.float64,
    )


def make_kalman_translation() -> cv2.KalmanFilter:
    """6 维状态 [x,y,z,vx,vy,vz]，观测 3 维平移；简单匀速模型。"""
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
    return kf


def make_kalman_bbox() -> cv2.KalmanFilter:
    """8 维状态 [cx,cy,w,h,vx,vy,vw,vh]，观测 4 维 [cx,cy,w,h]。"""
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
    kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
    return kf


def xyxy_to_cxcywh(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.tolist()
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)


def cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh.tolist()
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _parse_udp(addr: str) -> Optional[Tuple[str, int]]:
    if not addr or not addr.strip():
        return None
    host, _, port_s = addr.rpartition(":")
    if not host:
        raise ValueError("--udp 格式应为 host:port，例如 127.0.0.1:5005")
    return host.strip(), int(port_s)


def run(opt):
    device = select_device(opt.device)
    half = device.type != "cpu"

    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    names = model.module.names if hasattr(model, "module") else model.names

    nkpt = int(model.yaml.get("nkpt", 0) or 0)
    use_kpt = nkpt > 0
    if use_kpt and nkpt != 4:
        print(f"警告: 模型 nkpt={nkpt}，PnP 仍按前 4 个点使用。")

    if half:
        model.half()

    # 相机
    cap = cv2.VideoCapture(opt.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.frame_h)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("无法打开摄像头")
    h0, w0 = frame0.shape[:2]

    if opt.calib:
        camera_matrix, dist_coeffs = load_calib_yaml(opt.calib)
    elif opt.fx is not None:
        camera_matrix = build_camera_matrix_intrinsics(
            w0, h0, opt.fx, opt.fy, opt.cx, opt.cy
        )
        dist_coeffs = np.zeros(5, dtype=np.float64)
    else:
        camera_matrix = build_camera_matrix(w0, h0, opt.fov_deg)
        dist_coeffs = np.zeros(5, dtype=np.float64)

    kf = make_kalman_translation()
    kf_initialized = False
    bbox_kf = make_kalman_bbox()
    bbox_kf_initialized = False
    bbox_missed = 0
    last_t = time.time()
    last_stream_t = 0.0
    frame_idx = 0

    udp_target = _parse_udp(getattr(opt, "udp", "") or "")
    udp_sock = None
    if udp_target:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    if use_kpt:
        print("按 q 退出。模式：关键点模型。")
    else:
        print("按 q 退出。模式：纯检测权重 — 使用检测框四角做 PnP")
    if getattr(opt, "no_window", False):
        print(
            "无界面模式：用 Ctrl+C 结束。"
            + (
                ""
                if getattr(opt, "stream_json", False)
                else " 建议同时加 --stream-json 否则无输出。"
            ),
            flush=True,
        )

    while True:
        ok, im0 = cap.read()
        if not ok:
            break
        im0 = im0.copy()
        frame_idx += 1
        img, ratio, pad = letterbox(im0, new_shape=imgsz, stride=stride, auto=True)
        img = img.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        img = np.ascontiguousarray(img)
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img_t)[0]
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=False,
            kpt_label=use_kpt,
            nc=model.yaml["nc"],
            nkpt=nkpt if use_kpt else None,
        )
        t2 = time_synchronized()

        now = time.time()
        dt = max(now - last_t, 1e-3)
        last_t = now
        bbox_pred_xyxy = None
        if bbox_kf_initialized:
            bbox_kf.transitionMatrix[0, 4] = dt
            bbox_kf.transitionMatrix[1, 5] = dt
            bbox_kf.transitionMatrix[2, 6] = dt
            bbox_kf.transitionMatrix[3, 7] = dt
            pred_state = bbox_kf.predict()[:4].reshape(-1)
            bbox_pred_xyxy = clip_xyxy(cxcywh_to_xyxy(pred_state), im0.shape[1], im0.shape[0])

        det = pred[0]
        label_txt = f"FPS:{1.0 / (t2 - t1 + 1e-9):.1f}"
        have_measurement = len(det) > 0
        have_box_for_pnp = False
        xyxy = None
        cls_id = -1
        conf = 0.0
        pnp_mode = "none"

        if have_measurement:
            det = det.clone()
            scale_coords(img_t.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            if use_kpt:
                scale_coords(
                    img_t.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3
                )

            # 取置信度最高的一个装甲
            best = det[det[:, 4].argmax()]
            xyxy = best[:4].cpu().numpy().astype(np.float32)
            cls_id = int(best[5].item())
            conf = float(best[4].item())
            pnp_mode = "detect"

            # bbox 跟踪：有检测就校正
            meas_box = xyxy_to_cxcywh(xyxy).reshape(4, 1)
            if not bbox_kf_initialized:
                bbox_kf.statePost[:4] = meas_box
                bbox_kf.statePost[4:] = 0
                bbox_kf_initialized = True
            else:
                bbox_kf.correct(meas_box)
            bbox_missed = 0
            have_box_for_pnp = True

            c = (0, 255, 0)  # 检测框
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(im0, (x1, y1), (x2, y2), c, 2)
            img_pts = None
            if use_kpt:
                kpt = best[6:].cpu().numpy().reshape(nkpt, 3)
                for ki in range(min(4, nkpt)):
                    px, py, _ = kpt[ki]
                    cv2.circle(im0, (int(px), int(py)), 4, (0, 165, 255), -1)
                img_pts = np.array(
                    [[kpt[i, 0], kpt[i, 1]] for i in range(4)], dtype=np.float64
                )
            else:
                for px, py in bbox_to_img_points_four_corners(xyxy):
                    cv2.circle(im0, (int(px), int(py)), 4, (0, 165, 255), -1)
                img_pts = bbox_to_img_points_four_corners(xyxy)
        elif bbox_pred_xyxy is not None:
            # 无检测：用预测框“续帧”
            bbox_missed += 1
            if bbox_missed <= opt.max_missed:
                xyxy = bbox_pred_xyxy
                have_box_for_pnp = True
                pnp_mode = "predict"
                c = (0, 215, 255)  # 预测框
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(im0, (x1, y1), (x2, y2), c, 2)
                cv2.putText(
                    im0,
                    f"PRED {bbox_missed}/{opt.max_missed}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    c,
                    2,
                )
                img_pts = bbox_to_img_points_four_corners(xyxy)
            else:
                bbox_kf_initialized = False
                bbox_missed = 0

        if have_box_for_pnp:
            ok_pnp, rvec, tvec = cv2.solvePnP(
                SMALL_ARMOR_POINTS_3D,
                img_pts,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok_pnp:
                dist_m = float(np.linalg.norm(tvec))
                name_text = names[cls_id] if cls_id >= 0 and cls_id < len(names) else "track"
                conf_text = f"{conf:.2f}" if cls_id >= 0 else "--"
                label_txt += f" | {name_text} {conf_text} | {pnp_mode} | dist~{dist_m:.2f}m"

                # 卡尔曼：用 tvec 作观测，时间步长用帧间隔（遮挡期间仅 predict）
                kf.transitionMatrix[0, 3] = dt
                kf.transitionMatrix[1, 4] = dt
                kf.transitionMatrix[2, 5] = dt
                if pnp_mode == "detect":
                    meas = tvec.astype(np.float32).reshape(3, 1)
                    if not kf_initialized:
                        kf.statePost[:3] = meas
                        kf.statePost[3:] = 0
                        kf_initialized = True
                    else:
                        kf.predict()
                        kf.correct(meas)
                else:
                    if kf_initialized:
                        kf.predict()
                t_smooth = kf.statePost[:3].ravel() if kf_initialized else tvec.ravel()
                dist_kf = float(np.linalg.norm(t_smooth))
                label_txt += f" | KF dist {dist_kf:.2f}m"

                if getattr(opt, "stream_json", False):
                    now = time.time()
                    interval = float(getattr(opt, "stream_interval", 0.0) or 0.0)
                    if now - last_stream_t >= interval:
                        last_stream_t = now
                        payload = {
                            "t": now,
                            "frame": frame_idx,
                            "inf_ms": round((t2 - t1) * 1000, 2),
                            "fps_inf": round(1.0 / (t2 - t1 + 1e-9), 2),
                            "mode": "kpt" if use_kpt else "bbox",
                            "pnp_mode": pnp_mode,
                            "class_id": cls_id,
                            "class_name": names[cls_id] if cls_id >= 0 and cls_id < len(names) else "track",
                            "conf": round(conf, 4),
                            "tvec_m": [float(x) for x in tvec.ravel()],
                            "rvec_rad": [float(x) for x in rvec.ravel()],
                            "dist_m": round(dist_m, 4),
                            "tvec_kf_m": [float(x) for x in t_smooth],
                            "dist_kf_m": round(dist_kf, 4),
                        }
                        line = json.dumps(payload, ensure_ascii=False)
                        print(line, flush=True)
                        if udp_sock and udp_target:
                            udp_sock.sendto((line + "\n").encode("utf-8"), udp_target)

        cv2.putText(
            im0,
            label_txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        if not getattr(opt, "no_window", False):
            cv2.imshow("armor PnP + KF (bbox or kpt)", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.001)

    cap.release()
    if not getattr(opt, "no_window", False):
        cv2.destroyAllWindows()
    if udp_sock:
        udp_sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="模型权重，如 armor_detect_12 的 best.pt 或四点关键点 best.pt",
    )
    parser.add_argument("--device", default="0", help="cuda:0 或 cpu")
    parser.add_argument("--img-size", type=int, default=416, help="与训练/导出一致")
    parser.add_argument("--conf-thres", type=float, default=0.4)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--frame-w", type=int, default=1280)
    parser.add_argument("--frame-h", type=int, default=720)
    parser.add_argument(
        "--calib",
        type=str,
        default="",
        help="标定 yaml：camera_matrix 3x3、dist_coeffs；优先级高于 --fx",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="水平焦距 fx（像素）。笔记本可先用分辨率 + --fov-deg 估，或标定后填入",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="垂直焦距 fy（像素），默认与 fx 相同",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="主点 u（像素），默认宽的一半",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="主点 v（像素），默认高的一半",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=55.0,
        help="未使用 --calib 且未指定 --fx 时，用水平视场角估 fx（常见 50~70）",
    )
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="只保留指定类别 id")
    parser.add_argument(
        "--stream-json",
        action="store_true",
        help="每帧成功 PnP 后向 stdout 输出一行 JSON（实时，可管道给其它程序）",
    )
    parser.add_argument(
        "--stream-interval",
        type=float,
        default=0.0,
        help="JSON 输出最小间隔（秒），0 表示每次有结果都输出",
    )
    parser.add_argument(
        "--udp",
        type=str,
        default="",
        metavar="HOST:PORT",
        help="同时把 JSON 行用 UDP 发到该地址，例如 127.0.0.1:5005",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="不显示窗口（仅推理 + 流式输出，适合无头采集）",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=12,
        help="检测丢失后仍显示预测框的最大连续帧数",
    )
    opt = parser.parse_args()
    run(opt)
