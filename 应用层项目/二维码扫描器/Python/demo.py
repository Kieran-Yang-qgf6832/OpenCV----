# -*- coding: utf-8 -*-
"""二维码扫描器：单张图精确解码 + 视频逐帧扫描，并把解码结果实时叠在画面上。

思路：
    1. QRCodeDetector.detectAndDecode 一次拿到内容、四角与矫正后的码图；
    2. 单张图展示四角与解码内容，并输出一行“扫描器”风格的取景框；
    3. 视频里逐帧检测，成功时画绿框并显示文本，失败时画搜索框；
    4. 逐帧写入 output/qr_scan.mp4，并统计成功率。

核心函数：cv2.QRCodeDetector.detectAndDecode / detect
输出：output/qr_detected.png、qr_scan.mp4、qr_frame.png
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


def panel(img, lines, color=(0, 255, 0)):
    """在图像下方追加一块黑底文字区，避免文字压住二维码。"""
    bar = np.zeros((34 * len(lines) + 12, img.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (12, 26 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return np.vstack([img, bar])


def scan_still(qr, img):
    """单张图：解码 + 画角点 + 拼信息栏。"""
    data, points, _ = qr.detectAndDecode(img)
    canvas = img.copy()
    if points is not None:
        pts = points.reshape(-1, 2).astype(int)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)
        for i, p in enumerate(pts):
            cv2.circle(canvas, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(canvas, "P%d" % i, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    lines = ["decoded: %s" % (data if data else "<nothing>"),
             "payload length: %d bytes" % len(data.encode("utf-8"))]
    return panel(canvas, lines), data


def scan_video(qr, path, display=False):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, hit, last_data = [], 0, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        data, points, _ = qr.detectAndDecode(frame)
        if points is not None:
            cv2.polylines(frame, [points.reshape(-1, 2).astype(int)], True, (0, 255, 0), 3)
        if data:
            hit += 1
            last_data = data
            cv2.putText(frame, "OK  %s" % data[:28], (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "searching...", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
        frames.append(frame)
    cap.release()
    print("视频 %d 帧，解码成功 %d 帧；内容：%s" % (len(frames), hit, last_data))
    write_video(out_path("qr_scan.mp4"), frames, fps)
    return frames


def main():
    ap = argparse.ArgumentParser(description="二维码检测与解码")
    ap.add_argument("--image", default=os.path.join(ASSETS, "qrcode.png"))
    ap.add_argument("--video", default=os.path.join(ASSETS, "scan.mp4"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    qr = cv2.QRCodeDetector()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit("读不到图片：%s" % args.image)
    still, data = scan_still(qr, img)
    if not data:
        data2, _, _ = qr.detectAndDecodeCurved(img) if hasattr(qr, "detectAndDecodeCurved") else ("", None, None)
        data = data2 or data
    print("单张图解码结果：%r（%d 字节）" % (data, len(data.encode())))
    cv2.imwrite(out_path("qr_detected.png"), still)

    if os.path.exists(args.video):
        frames = scan_video(qr, args.video, args.display)
        if frames:
            cv2.imwrite(out_path("qr_frame.png"), frames[len(frames) // 2])
    print("已保存 output/qr_detected.png、qr_scan.mp4、qr_frame.png")

    if args.display:
        cv2.imshow("qr", still)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
