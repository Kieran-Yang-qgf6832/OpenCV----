# -*- coding: utf-8 -*-
"""视频稳像：估计逐帧全局位移，平滑运动轨迹后反向补偿，输出稳像视频。

思路：
    1. 灰度相位相关（phaseCorrelate）估出每帧的累计平移轨迹；
    2. 对轨迹做滑动平均得到“理想轨迹”，差值即为需要补偿的抖动；
    3. 用仿射平移把每帧拉回理想轨迹，并裁掉边缘无效区；
    4. 左右并排输出原始/稳像视频，并画抖动曲线前后的对比图。

核心函数：cv2.phaseCorrelate / cv2.warpAffine
输出：output/stabilized.mp4、stabilize_compare.png、stabilize_curve.png
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

SMOOTH_RADIUS = 15
CROP_RATIO = 0.03


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


def smooth_curve(curve, radius):
    """滑动平均：边界按可用窗口取均值，避免两端塌陷。"""
    out = np.empty_like(curve)
    n = len(curve)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out[i] = curve[lo:hi].mean(axis=0)
    return out


def draw_curves(raw, smooth, size=(1000, 320)):
    """画原始轨迹与理想轨迹（两张子图），并标注抖动幅度下降。"""
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, "trajectory: raw (red) vs smoothed (green)", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    pad = 40
    lo = min(raw.min(), smooth.min())
    hi = max(raw.max(), smooth.max())
    span = max(hi - lo, 1e-6)
    for curve, color in ((raw, (40, 60, 220)), (smooth, (40, 170, 40))):
        pts = [(int(pad + i * (w - 2 * pad) / max(len(curve) - 1, 1)),
                int(h - pad - (curve[i, 0] - lo) * (h - 2 * pad) / span)) for i in range(len(curve))]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 2)
    cv2.line(canvas, (pad, h - pad), (w - pad, h - pad), (0, 0, 0), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="基于全局平移估计的视频稳像")
    ap.add_argument("--video", default=os.path.join(ASSETS, "shaky.mp4"))
    ap.add_argument("--radius", type=int, default=SMOOTH_RADIUS)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        sys.exit("帧数太少")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    shifts = np.zeros((len(frames), 2), np.float32)
    for i in range(1, len(frames)):
        (dx, dy), response = cv2.phaseCorrelate(grays[i - 1], grays[i])
        shifts[i] = shifts[i - 1] + (dx, dy)
    smooth = smooth_curve(shifts, args.radius)
    # 抖动强度用“帧间位移的波动”衡量：补偿后帧间位移应当贴近平滑轨迹的帧间位移
    jitter_before = np.diff(shifts, axis=0).std(axis=0)
    jitter_after = np.diff(smooth, axis=0).std(axis=0)
    print("帧间抖动 std：补偿前 x=%.2f / y=%.2f px，补偿后 x=%.2f / y=%.2f px（下降 %.0f%%）"
          % (jitter_before[0], jitter_before[1], jitter_after[0], jitter_after[1],
             100.0 * (1 - jitter_after.mean() / max(jitter_before.mean(), 1e-6))))

    h, w = frames[0].shape[:2]
    crop = int(min(h, w) * CROP_RATIO)
    output = []
    for i, f in enumerate(frames):
        dx, dy = (smooth[i] - shifts[i]).round().astype(int)
        M = np.float32([[1, 0, dx + crop], [0, 1, dy + crop]])
        warped = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        left = f[crop:h - crop, crop:w - crop]
        right = warped[crop:h - crop, crop:w - crop]
        pair = np.hstack([left, right])
        cv2.putText(pair, "original", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(pair, "stabilized", (left.shape[1] + 14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        output.append(pair)
    write_video(out_path("stabilized.mp4"), output, fps)

    cv2.imwrite(out_path("stabilize_compare.png"), output[len(output) // 2])
    cv2.imwrite(out_path("stabilize_curve.png"), draw_curves(shifts, smooth))
    print("已保存 output/stabilized.mp4、stabilize_compare.png、stabilize_curve.png")

    if args.display:
        for f in output:
            cv2.imshow("stabilize", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
