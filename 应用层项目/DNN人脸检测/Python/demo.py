# -*- coding: utf-8 -*-
"""DNN 人脸检测：用 SSD(res10) 做多尺度人脸检测，输出框、置信度与检出统计。

思路：
    1. 加载 res10_300x300_ssd 人脸检测网络（Caffe），输入按训练均值归一化；
    2. 前向得到每个人脸框与置信度，做一次 NMS 去掉重叠框；
    3. 在人像大图与拼贴合影上分别跑一遍，验证多人脸场景；
    4. 依次尝试多尺度输入（300/600），对比小脸检出情况。

核心函数：cv2.dnn.readNetFromCaffe / cv2.dnn.NMSBoxes
输出：output/faces_detected.png、faces_group.png
"""

import argparse
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖：pip install opencv-contrib-python numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ASSETS = os.path.join(PROJ, "assets")
OUT = os.path.join(PROJ, "output")
MODELS = os.path.join(os.path.dirname(PROJ), "_models")

MEAN = (104.0, 177.0, 123.0)
CONF = 0.6


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_net():
    proto = os.path.join(MODELS, "res10_deploy.prototxt")
    model = os.path.join(MODELS, "res10_face.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(model)):
        sys.exit("缺少模型：%s / %s（见 README 的模型下载说明）" % (proto, model))
    return cv2.dnn.readNetFromCaffe(proto, model)


def detect_faces(net, img, conf=CONF, input_size=300):
    """返回 [(x, y, w, h, score), ...]，已做 NMS。"""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (input_size, input_size)), 1.0,
                                 (input_size, input_size), MEAN)
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        score = float(det[0, 0, i, 2])
        if score < conf:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.35)
    keep = np.array(keep).ravel() if len(keep) else []
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in keep]


def draw(img, faces, tag):
    canvas = img.copy()
    for i, (x, y, w, h, score) in enumerate(faces):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(canvas, "#%d %.2f" % (i, score), (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(canvas, (x + w // 2, y + h // 2), 4, (0, 0, 255), -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(canvas, "%s  faces: %d" % (tag, len(faces)), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="res10 SSD 人脸检测")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    net = load_net()

    single = cv2.imread(os.path.join(ASSETS, "faces.png"))
    if single is None:
        sys.exit("缺少素材 assets/faces.png")
    faces = detect_faces(net, single, args.conf)
    print("单人像：检出 %d 张脸，置信度 %s" % (len(faces), ["%.3f" % f[4] for f in faces]))
    cv2.imwrite(out_path("faces_detected.png"), draw(single, faces, "single portrait"))

    group_path = os.path.join(ASSETS, "group_photo.png")
    if os.path.exists(group_path):
        group = cv2.imread(group_path)
        gfaces = detect_faces(net, group, args.conf)
        print("拼贴合影：检出 %d 张脸，置信度 %s" % (len(gfaces), ["%.3f" % f[4] for f in gfaces]))
        for size in (300, 600):
            alt = detect_faces(net, group, args.conf, input_size=size)
            print("  输入 %dpx 时检出 %d 张（小脸对输入尺度敏感）" % (size, len(alt)))
        cv2.imwrite(out_path("faces_group.png"), draw(group, gfaces, "group photo"))

    print("已保存 output/faces_detected.png" + ("、faces_group.png" if os.path.exists(group_path) else ""))

    if args.display:
        cv2.imshow("faces", cv2.imread(out_path("faces_detected.png")))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
