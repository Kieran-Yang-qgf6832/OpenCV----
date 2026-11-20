# -*- coding: utf-8 -*-
"""实时绿幕抠图：HSV 色度键提取前景，抑制绿边后合成到新背景，逐帧输出。

思路：
    1. HSV 里绿色有稳定的色相区间，据此得到背景掩膜；
    2. 对掩膜做开/闭运算与一次高斯羽化，边缘不会有锯齿；
    3. 前景做“绿边抑制”：把绿色通道限制在红蓝通道的最大值，去掉溢出的绿光；
    4. alpha 混合贴到目标背景，左右并排写入视频。

核心函数：cv2.inRange / cv2.morphologyEx / cv2.GaussianBlur / cv2.add
输出：output/green_key.mp4、green_frame.png、green_mask.png
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

HUE_LO, HUE_HI = 35, 95      # 绿色色相区间
SAT_MIN = 90
VAL_MIN = 40


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


def key_mask(frame):
    """绿色背景掩膜（255 = 背景）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (HUE_LO, SAT_MIN, VAL_MIN), (HUE_HI, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.GaussianBlur(mask, (5, 5), 0)


def desaturate_green(frame):
    """绿边抑制：绿色通道不得超过红蓝通道的较大值，去除反光溢出的绿色。"""
    b, g, r = cv2.split(frame.astype(np.float32))
    limit = np.maximum(b, r)
    g = np.minimum(g, limit)
    return cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)


def composite(frame, background, feather=9):
    mask = key_mask(frame)
    alpha = cv2.GaussianBlur(255 - mask, (feather | 1, feather | 1), 0).astype(np.float32) / 255.0
    alpha3 = cv2.merge([alpha, alpha, alpha])
    fg = desaturate_green(frame).astype(np.float32)
    bg = background.astype(np.float32)
    out = fg * alpha3 + bg * (1 - alpha3)
    return out.clip(0, 255).astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser(description="HSV 色度键绿幕抠图")
    ap.add_argument("--video", default=os.path.join(ASSETS, "green_screen.mp4"))
    ap.add_argument("--background", default=os.path.join(ASSETS, "background.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bg = cv2.imread(args.background)
    if bg is None:
        sys.exit("读不到背景：%s" % args.background)

    frames, keyed, masks = [], 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        comp, mask = composite(frame, bg)
        keyed += int((mask > 0).mean() * 100)
        if len(masks) < 1:
            masks.append(mask)
        pair = np.hstack([frame, comp])
        cv2.putText(pair, "original", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(pair, "composited", (frame.shape[1] + 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frames.append(pair)
    cap.release()
    if not frames:
        sys.exit("没有读到帧")

    print("处理 %d 帧，平均抠掉背景 %.0f%% 的面积" % (len(frames), keyed / len(frames)))
    write_video(out_path("green_key.mp4"), frames, fps)
    cv2.imwrite(out_path("green_frame.png"), frames[len(frames) // 2])
    if masks:
        cv2.imwrite(out_path("green_mask.png"), masks[0])
    print("已保存 output/green_key.mp4、green_frame.png、green_mask.png")

    if args.display:
        for f in frames:
            cv2.imshow("green key", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
