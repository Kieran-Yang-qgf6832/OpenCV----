# -*- coding: utf-8 -*-
"""人脸识别（小底库 1:N）：LBPH 训练两个人的样本库，再识别未参与训练的探针照片。

流程：
    1. res10 SSD 自动定位并裁剪人脸，灰度化 + 直方图均衡，统一到 128x128；
    2. 每个人用 1 张注册照做增强（镜像/旋转/亮度）得到 6 个训练样本，训练 LBPH；
    3. 探针是未参与训练的扰动版本（旋转 + 亮度变化），逐张报出“判给谁 + 距离”；
    4. 距离大于阈值判为“库外人员”，并把结论画成一张对比图。

已知局限：
    LBPH 是纹理特征，长相接近的两个人（本素材里的两张人像就属于这种）距离会拉不开；
    真要可靠拒识需要人脸 embedding（例如 FaceNet/ArcFace 的 ONNX），属于超纲的外部模型。

核心函数：cv2.face.LBPHFaceRecognizer_create / predict
输出：output/face_recognize.png
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

FACE_SIZE = (128, 128)
THRESHOLD = 70.0   # LBPH 距离阈值：超过则判为库外人员
MEAN = (104.0, 177.0, 123.0)


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_detector():
    proto = os.path.join(MODELS, "res10_deploy.prototxt")
    model = os.path.join(MODELS, "res10_face.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(model)):
        sys.exit("缺少人脸检测模型：_models/res10_*.caffemodel（见 README）")
    return cv2.dnn.readNetFromCaffe(proto, model)


def detect_largest(net, img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), MEAN)
    net.setInput(blob)
    det = net.forward()
    best = None
    for i in range(det.shape[2]):
        if float(det[0, 0, i, 2]) < 0.5:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        if best is None or (x2 - x1) * (y2 - y1) > (best[2] - best[0]) * (best[3] - best[1]):
            best = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    return best


def crop_face(net, img, pad=0.18):
    """裁出人脸并做灰度 + 直方图均衡，降低光照差异对 LBPH 的影响。"""
    box = detect_largest(net, img)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * w))
    y1 = max(0, int(y1 - pad * h))
    x2 = min(img.shape[1], int(x2 + pad * w))
    y2 = min(img.shape[0], int(y2 + pad * h))
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.resize(gray, FACE_SIZE))


def augment(face):
    """注册样本增强：镜像 + 旋转 ±10° + 亮度 ±15%。"""
    variants = [face, cv2.flip(face, 1)]
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), angle, 1.0)
        variants.append(cv2.warpAffine(face, M, FACE_SIZE, borderMode=cv2.BORDER_REPLICATE))
    for gain in (0.85, 1.15):
        variants.append(np.clip(face.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    return variants


def held_out(face, angle, gain):
    """训练集里没出现过的扰动，用来当探针。"""
    M = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), angle, 1.0)
    v = cv2.warpAffine(face, M, FACE_SIZE, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(v.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="LBPH 小底库人脸识别")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    net = load_detector()
    gallery = [("ID-A", os.path.join(ASSETS, "face.png")),
               ("ID-B", os.path.join(PROJ, "..", "实时姿态估计", "assets", "person.png"))]
    probe_src = os.path.join(ASSETS, "probe.png")

    faces, labels, names = [], [], []
    previews = []
    for idx, (name, path) in enumerate(gallery):
        img = cv2.imread(os.path.abspath(path))
        if img is None:
            sys.exit("缺少素材：%s" % path)
        face = crop_face(net, img)
        if face is None:
            sys.exit("%s 里没检出人脸" % name)
        faces += augment(face)
        labels += [idx] * len(augment(face))
        names.append(name)
        previews.append(face)

    model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    model.train(faces, np.array(labels, np.int32))
    print("底库：%s，每人 %d 个训练样本，合计 %d 张" % ("、".join(names), len(augment(faces[0])), len(faces)))

    probes = []
    for idx, (name, face) in enumerate(zip(names, previews)):
        for angle, gain in ((6, 1.0), (14, 0.9), (-10, 1.1)):
            probes.append(("%s probe rot%+d" % (name, angle), held_out(face, angle, gain), idx))
    probe_img = cv2.imread(probe_src)
    if probe_img is not None:
        other = crop_face(net, probe_img)
        if other is not None:
            probes.append(("UNKNOWN", held_out(other, 6, 1.0), -1))

    results, correct = [], 0
    for tag, probe, truth in probes:
        label, dist = model.predict(probe)
        accepted = dist < args.threshold
        who = names[label] if accepted else "UNKNOWN"
        ok = (truth >= 0 and accepted and label == truth) or (truth < 0 and not accepted)
        correct += int(ok)
        results.append((tag, who, float(dist), ok, probe))
        print("  %-18s -> %-6s 距离 %6.2f  %s" % (tag, who, dist, "OK" if ok else "错"))

    # 4 列网格：每格一张探针人脸 + 判定结果
    tile, card_h, cols = 180, 250, 4
    rows = (len(results) + cols - 1) // cols
    head = 44
    canvas = np.zeros((head + rows * card_h, cols * tile, 3), np.uint8)
    cv2.putText(canvas, "1:N LBPH  %d/%d correct  threshold %.0f" % (correct, len(results), args.threshold),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    for i, (tag, who, dist, ok, probe) in enumerate(results):
        r, c = divmod(i, cols)
        y0, x0 = head + r * card_h, c * tile
        canvas[y0 : y0 + tile, x0 : x0 + tile] = cv2.cvtColor(cv2.resize(probe, (tile, tile)), cv2.COLOR_GRAY2BGR)
        color = (0, 220, 0) if ok else (0, 0, 255)
        cv2.rectangle(canvas, (x0, y0), (x0 + tile - 1, y0 + card_h - 1), color, 2)
        cv2.putText(canvas, who, (x0 + 8, y0 + tile + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(canvas, "d=%.1f" % dist, (x0 + 8, y0 + tile + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(canvas, tag[:15], (x0 + 8, y0 + tile + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imwrite(out_path("face_recognize.png"), canvas)
    print("识别正确 %d / %d，阈值 %.0f；结果见 output/face_recognize.png" % (correct, len(results), args.threshold))

    if args.display:
        cv2.imshow("recognize", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
