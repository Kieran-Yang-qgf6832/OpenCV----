# -*- coding: utf-8 -*-
"""AR 标记叠加：由 ArUco 标定板解出相机位姿，把虚拟立方体“立”在板上并跟随相机。

思路：
    1. 每帧检测 3x3 标定板上的 9 个标记，按 id 拼出板坐标系下的 3D 角点；
    2. solvePnP 解出板相对相机的位姿，并用上一帧结果做指数平滑抑制抖动；
    3. 把虚拟立方体的 8 个顶点投影回图像，画成带明暗面的实体方块；
    4. 逐帧写入 output/ar_overlay.mp4。

核心函数：cv2.aruco.detectMarkers / cv2.solvePnP / cv2.projectPoints / cv2.fillConvexPoly
输出：output/ar_overlay.mp4、ar_frame.png
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

MARKER_MM = 40.0
GAP_MM = 12.0
GRID = 3
CUBE_H = 70.0


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def write_video(path, frames, fps=25.0):
    """mp4v 先写临时文件，再用 ffmpeg 转 H.264（浏览器/文档都能播）。"""
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


def board_points(marker_mm, gap_mm, grid):
    step = marker_mm + gap_mm
    half = marker_mm / 2.0
    table = {}
    for mid in range(grid * grid):
        row, col = divmod(mid, grid)
        cx, cy = col * step, -row * step
        table[mid] = np.array([[cx - half, cy + half, 0], [cx + half, cy + half, 0],
                               [cx + half, cy - half, 0], [cx - half, cy - half, 0]], np.float32)
    return table


def cube_vertices(marker_mm, gap_mm, grid, height):
    span = (grid - 1) * (marker_mm + gap_mm) + marker_mm
    return np.float32([[0, 0, 0], [span, 0, 0], [span, -span, 0], [0, -span, 0],
                       [0, 0, -height], [span, 0, -height], [span, -span, -height], [0, -span, -height]])


def draw_cube(img, K, rvec, tvec, verts):
    proj, _ = cv2.projectPoints(verts, rvec, tvec, K, None)
    p = proj.reshape(-1, 2)
    top = np.int32([p[4], p[5], p[6], p[7]])
    left = np.int32([p[0], p[1], p[5], p[4]])
    right = np.int32([p[1], p[2], p[6], p[5]])
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, top, (200, 160, 40))
    cv2.fillConvexPoly(overlay, left, (150, 110, 30))
    cv2.fillConvexPoly(overlay, right, (110, 80, 20))
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(np.int32(p[a])), tuple(np.int32(p[b])), (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="ArUco 增强现实叠加")
    ap.add_argument("--input", default=os.path.join(ASSETS, "ar_scene.mp4"))
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM)
    ap.add_argument("--gap-mm", type=float, default=GAP_MM)
    ap.add_argument("--smooth", type=float, default=0.6, help="位姿平滑系数，0 表示不平滑")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    obj_table = board_points(args.marker_mm, args.gap_mm, GRID)
    cube = cube_vertices(args.marker_mm, args.gap_mm, GRID, CUBE_H)

    frames, smooth_r, smooth_t, hit = [], None, None, 0
    K = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if K is None:
            h, w = frame.shape[:2]
            f = 1.2 * w
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)

        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
        if ids is not None and len(ids) >= 4:
            obj_pts, img_pts = [], []
            for corner, mid in zip(corners, ids.ravel()):
                if int(mid) in obj_table:
                    obj_pts.append(obj_table[int(mid)])
                    img_pts.append(corner.reshape(4, 2).astype(np.float32))
            ok_pnp, rvec, tvec = cv2.solvePnP(np.vstack(obj_pts), np.vstack(img_pts), K, None,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_pnp:
                hit += 1
                if smooth_r is None or args.smooth <= 0:
                    smooth_r, smooth_t = rvec.copy(), tvec.copy()
                else:
                    a = args.smooth
                    smooth_r = a * smooth_r + (1 - a) * rvec
                    smooth_t = a * smooth_t + (1 - a) * tvec
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                draw_cube(frame, K, smooth_r, smooth_t, cube)
        frames.append(frame)
    cap.release()

    if not frames:
        sys.exit("视频里没有读到帧")

    print("处理帧数 %d，其中 %d 帧解出位姿（%.0f%%）" % (len(frames), hit, 100.0 * hit / len(frames)))
    write_video(out_path("ar_overlay.mp4"), frames, fps)
    mid = frames[len(frames) // 2]
    cv2.imwrite(out_path("ar_frame.png"), mid)
    print("已保存 output/ar_overlay.mp4、output/ar_frame.png")

    if args.display:
        for f in frames:
            cv2.imshow("ar", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
