# -*- coding: utf-8 -*-
"""目标跟踪：给定首帧框，用相关滤波跟踪器跟住运动目标，并用颜色分割当真值量化误差。

思路：
    1. 首帧按 init_box.txt 初始化 CSRT 跟踪器；
    2. 逐帧 update 得到新框，画出框与历史轨迹；
    3. 目标是很纯的红色，用颜色分割出的质心作为近似真值，逐帧算跟踪框中心误差；
    4. 输出跟踪视频 + 误差曲线 + 误差统计。

核心函数：cv2.TrackerCSRT_create / tracker.init / tracker.update
输出：output/tracked.mp4、track_frame.png、track_error.png
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


def red_centroid(img):
    """红色目标的质心，作为独立于跟踪器的近似真值。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def plot_errors(errors, size=(900, 300)):
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, "tracking center error (px)  mean %.1f  max %.1f"
                % (float(np.mean(errors)), float(np.max(errors))),
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    top = max(float(np.max(errors)) * 1.3, 5.0)
    pts = [(int(50 + i * (w - 90) / max(len(errors) - 1, 1)),
            int(h - 40 - err * (h - 90) / top)) for i, err in enumerate(errors)]
    cv2.line(canvas, (50, h - 40), (w - 30, h - 40), (0, 0, 0), 1)
    cv2.line(canvas, (50, h - 40), (50, 40), (0, 0, 0), 1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (200, 80, 30), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="CSRT 单目标跟踪")
    ap.add_argument("--video", default=os.path.join(ASSETS, "target.mp4"))
    ap.add_argument("--box", default=os.path.join(ASSETS, "init_box.txt"))
    ap.add_argument("--tracker", default="CSRT", choices=["CSRT", "KCF", "MIL"])
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    if not ok:
        sys.exit("视频是空的")

    bbox = tuple(int(v) for v in open(args.box).read().strip().split(","))
    maker = {"CSRT": cv2.TrackerCSRT_create, "KCF": cv2.TrackerKCF_create, "MIL": cv2.TrackerMIL_create}[args.tracker]
    tracker = maker()
    tracker.init(first, bbox)
    print("跟踪器 %s 初始化框 %s" % (args.tracker, bbox))

    frames, trail, errors = [], [], []
    frame_idx = 0
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = first
        ok_track, box = tracker.update(frame)
        center = None
        if ok_track:
            x, y, w, h = (int(v) for v in box)
            center = (x + w / 2, y + h / 2)
            trail.append((int(center[0]), int(center[1])))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            for a, b in zip(trail[-40:-1], trail[-39:]):
                cv2.line(frame, a, b, (0, 200, 255), 2)
        truth = red_centroid(frame)
        if truth is not None and center is not None:
            err = float(np.hypot(center[0] - truth[0], center[1] - truth[1]))
            errors.append(err)
            cv2.circle(frame, (int(truth[0]), int(truth[1])), 4, (255, 0, 255), -1)
            cv2.putText(frame, "err %.1fpx" % err, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        frames.append(frame)
        frame_idx += 1
    cap.release()

    print("跟踪 %d 帧，成功 %d 帧" % (frame_idx, len(trail)))
    if errors:
        print("中心误差：mean %.2f px, max %.2f px, >10px 占比 %.0f%%"
              % (np.mean(errors), np.max(errors), 100.0 * np.mean(np.array(errors) > 10)))
    write_video(out_path("tracked.mp4"), frames, fps)
    cv2.imwrite(out_path("track_frame.png"), frames[len(frames) // 2])
    if errors:
        cv2.imwrite(out_path("track_error.png"), plot_errors(errors))
    print("已保存 output/tracked.mp4、track_frame.png、track_error.png")

    if args.display:
        for f in frames:
            cv2.imshow("track", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
