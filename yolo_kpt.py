import cv2
import numpy as np
import openvino as ov


# ===== 参数（对应.h里的宏）=====
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.1
IMG_SIZE = 416
KPT_NUM = 5   # 根据模式改
CLS_NUM = 4


class Object:
    def __init__(self):
        self.rect = None
        self.label = 0
        self.prob = 0.0
        self.kpt = []


class YOLO_KPT:
    def __init__(self, model_path, device="CPU"):
        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()

    # ===== sigmoid =====
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # ===== letterbox =====
    def letter_box(self, img):
        h, w = img.shape[:2]
        r = min(IMG_SIZE / h, IMG_SIZE / w)

        new_w, new_h = int(w * r), int(h * r)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        dw = (IMG_SIZE - new_w) // 2
        dh = (IMG_SIZE - new_h) // 2

        canvas[dh:dh+new_h, dw:dw+new_w] = resized
        return canvas, (dw, dh, r)

    # ===== 坐标还原 =====
    def scale_box(self, box, padd, raw_w, raw_h):
        dw, dh, r = padd

        x, y, w, h = box
        x = (x - dw) / r
        y = (y - dh) / r
        w /= r
        h /= r

        x = max(min(x, raw_w - 1), 0)
        y = max(min(y, raw_h - 1), 0)

        return [x, y, w, h]

    # ===== 核心：解析YOLO输出 =====
    def generate_proposals(self, stride, feat):
        objects = []

        feat_h = IMG_SIZE // stride
        feat_w = IMG_SIZE // stride

        # anchor（简化版）
        anchors = np.array([
            [6,5], [9,7], [13,9],
            [18,15], [30,23], [46,37],
            [60,52], [94,56], [125,72]
        ])

        anchor_group = 0 if stride == 8 else (1 if stride == 16 else 2)

        for anchor in range(3):
            for i in range(feat_h):
                for j in range(feat_w):

                    offset = anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3)
                    offset += i * feat_w * (5 + CLS_NUM + KPT_NUM * 3)
                    offset += j * (5 + CLS_NUM + KPT_NUM * 3)

                    data = feat[offset: offset + (5 + CLS_NUM + KPT_NUM * 3)]

                    box_conf = self.sigmoid(data[4])
                    if box_conf < CONF_THRESHOLD:
                        continue

                    x, y, w, h = data[0:4]

                    # 解码
                    x = (self.sigmoid(x) * 2 - 0.5 + j) * stride
                    y = (self.sigmoid(y) * 2 - 0.5 + i) * stride
                    w = (self.sigmoid(w) * 2) ** 2 * anchors[anchor_group*3+anchor][0]
                    h = (self.sigmoid(h) * 2) ** 2 * anchors[anchor_group*3+anchor][1]

                    # 类别
                    cls_scores = self.sigmoid(data[5:5+CLS_NUM])
                    cls_id = np.argmax(cls_scores)
                    cls_prob = cls_scores[cls_id]

                    conf = box_conf * cls_prob
                    if conf < CONF_THRESHOLD:
                        continue

                    obj = Object()
                    obj.rect = [x - w/2, y - h/2, w, h]
                    obj.label = cls_id
                    obj.prob = conf

                    # 关键点
                    for k in range(KPT_NUM):
                        kx = data[5 + CLS_NUM + k*3]
                        ky = data[5 + CLS_NUM + k*3 + 1]

                        kx = (kx * 2 - 0.5 + j) * stride
                        ky = (ky * 2 - 0.5 + i) * stride

                        obj.kpt.append([kx, ky])

                    objects.append(obj)

        return objects

    # ===== 主流程 =====
    def detect(self, img):
        raw_h, raw_w = img.shape[:2]

        # 1️⃣ 预处理
        input_img, padd = self.letter_box(img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))[None]

        # 2️⃣ 推理
        outputs = self.compiled_model([input_img])

        # 3️⃣ 后处理
        proposals = []
        strides = [8, 16, 32]

        for i, stride in enumerate(strides):
            feat = outputs[i].reshape(-1)
            objs = self.generate_proposals(stride, feat)
            proposals.extend(objs)

        # 4️⃣ NMS
        boxes = []
        scores = []

        for obj in proposals:
            x, y, w, h = obj.rect
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(float(obj.prob))

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

        results = []
        for i in indices:
            obj = proposals[i]
            obj.rect = self.scale_box(obj.rect, padd, raw_w, raw_h)
            results.append(obj)

        return results