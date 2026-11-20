# -*- coding: utf-8 -*-
"""运动跟踪：用稠密光流分割出运动区域，并对最大运动目标做框选与轨迹记录。

思路：
    1. Farneback 稠密光流得到每像素的位移矢量；
    2. 幅度超阈值的像素视为“在动”，做形态学清理后取最大连通域；
    3. 在图上画出运动矢量场（稀疏抽样）与目标框、历史轨迹；
    4. 输出带标注的视频与关键帧。

核心函数：cv2.calcOpticalFlowFarneback / cv2.findContours / cv2.boundingRect
输出：output/motion_tracks.mp4、motion_frame.png
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

MAG_THRESH = 1.2   # 判为运动的位移阈值（像素/帧）
MIN_AREA = 1500    # 目标最小面积


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


def draw_flow_arrows(img, flow, step=40):
    """在网格点上画运动矢量，直观看出运动方向。"""
    h, w = img.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            if fx * fx + fy * fy < 4.0:
                continue
            end = (int(x + fx * 3), int(y + fy * 3))
            cv2.arrowedLine(img, (x, y), end, (0, 255, 255), 1, tipLength=0.35)


def main():
    ap = argparse.ArgumentParser(description="稠密光流运动分割与跟踪")
    ap.add_argument("--video", default=os.path.join(ASSETS, "motion_track.mp4"))
    ap.add_argument("--threshold", type=float, default=MAG_THRESH)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, prev = cap.read()
    if not ok:
        sys.exit("视频是空的")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frames, trail, centers = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])
        mask = (mag > args.threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(c)
                centers.append((x + w // 2, y + h // 2))
                trail.append((x + w // 2, y + h // 2))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.drawContours(frame, [c], -1, (0, 200, 255), 1)
                mean_v = np.mean(flow[mask > 0], axis=0) if (mask > 0).any() else (0, 0)
                cv2.putText(frame, "v=(%.1f, %.1f) px/frame" % (mean_v[0], mean_v[1]),
                            (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        draw_flow_arrows(frame, flow)
        for a, b in zip(trail[-60:-1], trail[-59:]):
            cv2.line(frame, a, b, (255, 0, 255), 3)
        cv2.putText(frame, "motion pixels: %.1f%%" % (100.0 * float((mask > 0).mean())),
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        frames.append(frame)
        prev_gray = gray
    cap.release()

    if not frames:
        sys.exit("没有可处理的帧")
    print("处理 %d 帧，检出运动目标的帧数 %d" % (len(frames), len(centers)))
    if len(centers) >= 2:
        print("轨迹位移：dx=%.0f px, dy=%.0f px" % (centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]))
    write_video(out_path("motion_tracks.mp4"), frames, fps)
    cv2.imwrite(out_path("motion_frame.png"), frames[len(frames) // 2])
    print("已保存 output/motion_tracks.mp4、motion_frame.png")

    if args.display:
        for f in frames:
            cv2.imshow("motion track", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
