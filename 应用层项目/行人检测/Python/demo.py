# -*- coding: utf-8 -*-
"""行人检测：用 OpenCV DNN 跑 MobileNet-SSD，逐帧检出 person 类别并计数。

思路：
    1. readNetFromCaffe 加载 MobileNet-SSD（20 类，person 是第 15 类）；
    2. 300x300 blob 前向，按置信度过滤 person，再按图像尺寸还原框；
    3. 画框、标置信度、统计每帧人数；
    4. 输出检测视频、关键帧与逐帧人数曲线。

核心函数：cv2.dnn.readNetFromCaffe / blobFromImage / net.forward
输出：output/pedestrians_detected.mp4、pedestrian_frame.png、pedestrian_count.png
"""

import argparse
import os
import subprocess
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

PERSON_CLASS = 15
CONF = 0.35


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def write_video(path, frames, fps=25.0):
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "22", path], check=False)
    if done.returncode == 0 and os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def load_net():
    proto = os.path.join(MODELS, "mobilenetssd_deploy.prototxt")
    model = os.path.join(MODELS, "mobilenetssd.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(model)):
        sys.exit("缺少模型：%s / %s（见 README 的模型下载说明）" % (proto, model))
    return cv2.dnn.readNetFromCaffe(proto, model)


def detect(net, frame, conf=CONF):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        score = float(det[0, 0, i, 2])
        if score < conf or int(det[0, 0, i, 1]) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        boxes.append((int(x1), int(y1), int(x2), int(y2), score))
    return boxes


def draw(frame, boxes):
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "person %.2f" % score, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, "count: %d" % len(boxes), (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def count_curve(counts, size=(1000, 260)):
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, "pedestrians per frame  max %d  mean %.2f"
                % (max(counts), float(np.mean(counts))), (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2)
    top = max(max(counts), 1)
    pts = [(int(46 + i * (w - 80) / max(len(counts) - 1, 1)),
            int(h - 40 - c * (h - 90) / top)) for i, c in enumerate(counts)]
    cv2.line(canvas, (46, h - 40), (w - 34, h - 40), (0, 0, 0), 1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (180, 90, 30), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="MobileNet-SSD 行人检测")
    ap.add_argument("--video", default=os.path.join(ASSETS, "pedestrians.mp4"))
    ap.add_argument("--image", default=os.path.join(ASSETS, "street.png"))
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    net = load_net()

    still = cv2.imread(args.image)
    if still is not None:
        boxes = detect(net, still, args.conf)
        print("静态图检出 %d 人" % len(boxes))
        cv2.imwrite(out_path("pedestrian_still.png"), draw(still.copy(), boxes))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, counts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect(net, frame, args.conf)
        counts.append(len(boxes))
        frames.append(draw(frame, boxes))
    cap.release()
    if not frames:
        sys.exit("没有读到帧")

    print("视频 %d 帧：人数 mean %.2f，max %d，检出率为 %.0f%%"
          % (len(frames), np.mean(counts), max(counts), 100.0 * np.mean(np.array(counts) > 0)))
    write_video(out_path("pedestrians_detected.mp4"), frames, fps)
    cv2.imwrite(out_path("pedestrian_frame.png"), frames[len(frames) // 2])
    cv2.imwrite(out_path("pedestrian_count.png"), count_curve(counts))
    print("已保存 output/pedestrians_detected.mp4、pedestrian_frame.png、pedestrian_count.png")

    if args.display:
        for f in frames:
            cv2.imshow("pedestrian", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
