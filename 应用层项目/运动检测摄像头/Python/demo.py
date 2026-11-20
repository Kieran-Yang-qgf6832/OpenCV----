# -*- coding: utf-8 -*-
"""运动检测摄像头：模拟监控场景，帧差法判定“有人经过”并触发告警。

思路：
    1. 相邻帧灰度差 + 阈值 + 膨胀，得到运动掩膜；
    2. 掩膜面积超过门限即触发告警，记录事件起止帧与时长（带滞回，避免闪烁）；
    3. 画面上叠加 REC 标记、告警横幅、事件计数与时间戳；
    4. 输出告警视频与事件时间线图。

核心函数：cv2.absdiff / cv2.threshold / cv2.findContours
输出：output/motion_detected.mp4、motion_frame.png、motion_events.png
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

DIFF_THRESH = 25      # 帧差阈值
AREA_TRIGGER = 400    # 触发告警的运动面积
HOLD_FRAMES = 6       # 告警滞回帧数


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


def event_timeline(active, fps, size=(1000, 220)):
    """把逐帧告警状态画成时间线，直观看出告警区间。"""
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, "motion alarm timeline (green = alarm on)", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    y0, y1 = 70, 150
    cv2.rectangle(canvas, (40, y0), (w - 30, y1), (230, 230, 230), -1)
    for i, on in enumerate(active):
        if not on:
            continue
        x = int(40 + i * (w - 70) / max(len(active) - 1, 1))
        cv2.line(canvas, (x, y0), (x, y1), (60, 170, 60), 3)
    cv2.putText(canvas, "0s", (40, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "%.1fs" % (len(active) / fps), (w - 110, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="帧差法运动检测与告警")
    ap.add_argument("--video", default=os.path.join(ASSETS, "motion.mp4"))
    ap.add_argument("--threshold", type=int, default=DIFF_THRESH)
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

    frames, active, events = [], [], []
    hold = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moving_area, boxes = 0, []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            moving_area += area
            boxes.append((x, y, w, h))

        if moving_area > AREA_TRIGGER:
            hold = HOLD_FRAMES
        alarm = hold > 0
        hold = max(0, hold - 1)
        active.append(alarm)
        if alarm != (active[-2] if len(active) > 1 else False):
            events.append((idx, "ON" if alarm else "OFF"))

        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, (28, 28), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (44, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if alarm:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)
            cv2.putText(frame, "MOTION ALARM", (frame.shape[1] // 2 - 170, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "t=%.2fs  area=%d" % (idx / fps, moving_area), (16, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        frames.append(frame)
        prev_gray = gray
    cap.release()

    if not frames:
        sys.exit("没有可处理的帧")
    on_frames = sum(1 for a in active if a)
    print("处理 %d 帧（%.1f 秒），告警帧 %d（占 %.0f%%），事件切换 %d 次"
          % (len(frames), len(frames) / fps, on_frames, 100.0 * on_frames / len(active), len(events)))
    print("事件序列：", events[:10], "..." if len(events) > 10 else "")

    write_video(out_path("motion_detected.mp4"), frames, fps)
    cv2.imwrite(out_path("motion_frame.png"), frames[len(frames) // 2])
    cv2.imwrite(out_path("motion_events.png"), event_timeline(active, fps))
    print("已保存 output/motion_detected.mp4、motion_frame.png、motion_events.png")

    if args.display:
        for f in frames:
            cv2.imshow("motion", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
