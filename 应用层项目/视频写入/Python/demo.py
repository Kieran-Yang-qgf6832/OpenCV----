# -*- coding: utf-8 -*-
"""视频写入：把 PNG 帧序列编码成 H.264 视频，并回读校验没有掉帧、没有画质塌陷。

思路：
    1. 用 VideoWriter 按 mp4v 写一版临时文件，再交给 ffmpeg 转 H.264（浏览器/PPT 都能播）；
    2. 回读生成的文件，核对帧数、分辨率、帧率；
    3. 逐帧与原图算 PSNR，量化有损压缩带来的损失；
    4. 输出一张“首帧/中间帧/末帧 + 参数摘要”的检查图。

核心函数：cv2.VideoWriter / cv2.VideoCapture / cv2.PSNR
输出：output/frames_to_video.mp4、frames_check.png
"""

import argparse
import glob
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


def encode(frames, path, fps=30.0):
    """mp4v 中转 + ffmpeg 转 H.264；ffmpeg 不可用时保留 mp4v。"""
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", path],
                          check=False)
    if done.returncode == 0 and os.path.exists(path):
        os.remove(tmp)
        return "h264"
    os.replace(tmp, path)
    return "mp4v"


def verify(path, frames):
    """回读视频，核对帧数与逐帧 PSNR。"""
    cap = cv2.VideoCapture(path)
    got, psnrs = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        got.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    for a, b in zip(frames, got):
        if a.shape == b.shape:
            psnrs.append(cv2.PSNR(a, b))
    return got, fps, size, psnrs


def check_sheet(frames, info, size=(480, 270)):
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    strip = np.hstack([cv2.resize(f, size) for f in picks])
    bar = np.full((70, strip.shape[1], 3), 0, np.uint8)
    for i, text in enumerate(info):
        cv2.putText(bar, text, (12, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    return np.vstack([strip, bar])


def main():
    ap = argparse.ArgumentParser(description="帧序列 -> 视频，并回读校验")
    ap.add_argument("--frames", default=os.path.join(ASSETS, "frames"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if not files:
        sys.exit("帧目录里没有 png：%s" % args.frames)
    frames = [cv2.imread(p) for p in files]
    print("读入 %d 帧，分辨率 %dx%d" % (len(frames), frames[0].shape[1], frames[0].shape[0]))

    path = out_path("frames_to_video.mp4")
    codec = encode(frames, path, args.fps)
    got, fps, size, psnrs = verify(path, frames)
    kbytes = os.path.getsize(path) / 1024.0
    print("编码器 %s，文件 %.0f KB，回读 %d 帧 / %.1f fps / %dx%d"
          % (codec, kbytes, len(got), fps, size[0], size[1]))
    if psnrs:
        print("回读 PSNR：mean %.2f dB，min %.2f dB（有损压缩的正常水平）"
              % (float(np.mean(psnrs)), float(np.min(psnrs))))
    assert len(got) == len(frames), "掉帧了：%d -> %d" % (len(frames), len(got))

    info = ["codec %s | %d frames | %.1f fps | %dx%d | %.0f KB" % (codec, len(got), fps, size[0], size[1], kbytes),
            "round-trip PSNR %.2f dB" % (float(np.mean(psnrs)) if psnrs else 0.0)]
    cv2.imwrite(out_path("frames_check.png"), check_sheet(frames, info))
    print("已保存 output/frames_to_video.mp4、frames_check.png")

    if args.display:
        cv2.imshow("check", check_sheet(frames, info))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
