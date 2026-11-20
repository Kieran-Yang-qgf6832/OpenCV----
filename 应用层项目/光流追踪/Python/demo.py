# -*- coding: utf-8 -*-
"""光流追踪：稀疏 Lucas-Kanade 追踪特征点，稠密 Farneback 显示整幅运动场。

思路：
    1. 首帧用 goodFeaturesToTrack 取角点；
    2. 相邻帧 LK 金字塔光流得到新位置，用前后向一致性剔除跟丢的点；
    3. 特征点留下彩色拖尾（越新越亮），并统计平均位移；
    4. 右手边同时给出 Farneback 稠密光流（HSV 编码方向与幅度）。

核心函数：cv2.goodFeaturesToTrack / cv2.calcOpticalFlowPyrLK / cv2.calcOpticalFlowFarneback
输出：output/flow_tracks.mp4、flow_frame.png、flow_dense.png
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

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
TRAIL = 25


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


def dense_flow_vis(prev_gray, gray):
    """Farneback 稠密光流：色相表示方向、明度表示速度。"""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), float(np.mean(mag))


def track(video, display=False):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    if not ok:
        sys.exit("视频是空的")

    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 12)
    trails = [[tuple(np.ravel(p))] for p in pts]
    frames, speeds = [], []
    frame, prev_gray = first, gray
    while True:
        if frames:
            ok, frame = cap.read()
            if not ok:
                break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(pts):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
            back, status_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
            keep = []
            old_pts = pts.reshape(-1, 2)
            new_pts = nxt.reshape(-1, 2)
            for i in range(len(old_pts)):
                fb_err = np.linalg.norm(old_pts[i] - back.reshape(-1, 2)[i])
                if status[i] == 1 and status_b[i] == 1 and fb_err < 1.0:
                    keep.append(i)
            if keep:
                speeds.append(float(np.mean(np.linalg.norm(new_pts[keep] - old_pts[keep], axis=1))))
            trails = [trails[i] for i in keep]
            pts = new_pts[keep].reshape(-1, 1, 2)
            for tr, p in zip(trails, pts.reshape(-1, 2)):
                tr.append(tuple(p))

        for tr in trails:
            tail = tr[-TRAIL:]
            for k, (a, b) in enumerate(zip(tail[:-1], tail[1:])):
                alpha = (k + 1) / max(len(tail), 1)
                color = (int(60 + 195 * alpha), int(255 - 120 * alpha), 40)
                cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)), color, 2)
            cv2.circle(frame, tuple(np.int32(tail[-1])), 2, (0, 255, 255), -1)

        dense, mean_mag = dense_flow_vis(prev_gray, gray)
        # 稠密光流放在右上角小窗里，与画面 50/50 混合，避免整块高饱和色盖住内容
        dense = cv2.resize(dense, (frame.shape[1] // 4, frame.shape[0] // 4))
        x0, y0 = frame.shape[1] - dense.shape[1] - 8, 8
        roi = frame[y0 : y0 + dense.shape[0], x0 : x0 + dense.shape[1]]
        frame[y0 : y0 + dense.shape[0], x0 : x0 + dense.shape[1]] = cv2.addWeighted(roi, 0.45, dense, 0.55, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + dense.shape[1], y0 + dense.shape[0]), (255, 255, 255), 1)
        cv2.putText(frame, "dense flow", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "tracked %d  mean |d|=%.2fpx" % (len(pts), speeds[-1] if speeds else 0.0),
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        frames.append(frame)
        prev_gray = gray
    cap.release()
    return frames, speeds, int(len(pts)), fps


def main():
    ap = argparse.ArgumentParser(description="稀疏+稠密光流追踪")
    ap.add_argument("--video", default=os.path.join(ASSETS, "flow.mp4"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    frames, speeds, kept, fps = track(args.video, args.display)
    if not frames:
        sys.exit("没有处理到任何帧")
    print("处理 %d 帧，末帧仍被稳定跟住的特征点 %d 个，平均位移 %.2f px/帧"
          % (len(frames), kept, float(np.mean(speeds)) if speeds else 0.0))
    write_video(out_path("flow_tracks.mp4"), frames, fps)
    cv2.imwrite(out_path("flow_frame.png"), frames[len(frames) // 2])
    cv2.imwrite(out_path("flow_last_frame.png"), frames[-1])
    print("已保存 output/flow_tracks.mp4、flow_frame.png、flow_last_frame.png")

    if args.display:
        for f in frames:
            cv2.imshow("flow", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
