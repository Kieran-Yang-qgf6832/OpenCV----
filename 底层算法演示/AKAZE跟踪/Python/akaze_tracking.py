# -*- coding: utf-8 -*-
"""
@file    akaze_tracking.py
@brief   AKAZE 跟踪演示：以首帧为参考，逐帧匹配并估计单应，绘制跟踪框。
@details 流程：初始化参考帧 -> 逐帧匹配 -> 估计变换 -> 绘制轨迹并写出视频。
@note    依赖：opencv-python、numpy
         匹配 + 单应比逐帧光流更抗尺度变化：本素材镜头在平移同时轻微缩小，
         用相似变换（单应）能把缩放一并吸收，框不会随时间漂移出画面。
"""

import argparse
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30


def locate_asset(file_name):
    """@brief 在若干候选位置中查找素材，便于从不同工作目录运行。"""
    candidates = [
        os.path.join("..", "..", "assets", file_name),
        os.path.join("..", "assets", file_name),
        os.path.join("assets", file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def output_dir_for(input_path):
    """@brief 由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。"""
    directory = os.path.dirname(input_path)
    if os.path.basename(directory) == "assets":
        directory = os.path.dirname(directory)
    return os.path.join(directory, "output")


def fit_panel(image, title):
    """@brief 等比缩放并居中留白到固定面板。"""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    scale = min(PANEL_WIDTH / bgr.shape[1], PANEL_HEIGHT / bgr.shape[0])
    scaled = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.full((PANEL_HEIGHT + TITLE_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    x = (PANEL_WIDTH - scaled.shape[1]) // 2
    y = TITLE_HEIGHT + (PANEL_HEIGHT - scaled.shape[0]) // 2
    panel[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def montage(panels, cols):
    """@brief 按列数拼接等尺寸面板。"""
    rows = (len(panels) + cols - 1) // cols
    h, w = panels[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        canvas[r * h:r * h + panel.shape[0], c * w:c * w + panel.shape[1]] = panel[:h, :w]
    return canvas


def track_quad(detector, ref_points, ref_desc, ref_corners, frame, ratio):
    """
    @brief  在待匹配帧上估计参考框的新位置。
    @param  detector   AKAZE 检测器。
    @param  ref_points 参考帧的关键点坐标。
    @param  ref_desc   参考帧描述子。
    @param  ref_corners 参考框的四个角。
    @param  frame      当前帧（灰度）。
    @param  ratio      Lowe 比率阈值。
    @return 当前帧的跟踪框；失败时返回 None。
    """
    keypoints, descriptors = detector.detectAndCompute(frame, None)
    if descriptors is None or len(descriptors) == 0:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(ref_desc, descriptors, k=2)
    src, dst = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            src.append(ref_points[pair[0].queryIdx])
            dst.append(keypoints[pair[0].trainIdx].pt)
    if len(src) < 8:
        return None

    homography, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 3.0)
    if homography is None:
        return None
    return cv2.perspectiveTransform(ref_corners, homography)


def main():
    """@brief 主流程：逐帧跟踪并输出视频与采样帧拼接图。"""
    parser = argparse.ArgumentParser(
        description="AKAZE 跟踪演示：首帧参考 + 逐帧匹配 + 单应估计")
    parser.add_argument("-i", "--input", default=None,
                        help="视频或首帧图像，缺省读取 assets/tracking.mp4")
    parser.add_argument("-o", "--output", default=None,
                        help="输出目录，缺省写入 <目录>/output")
    parser.add_argument("-r", "--ratio", type=float, default=0.75, help="Lowe 比率阈值")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("tracking.mp4")
    out_dir = args.output or output_dir_for(input_path)
    os.makedirs(out_dir, exist_ok=True)

    # 步骤 1：初始化参考帧（视频首帧）。
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise FileNotFoundError("无法打开视频: {}".format(os.path.abspath(input_path)))
    ok, reference = capture.read()
    if not ok:
        raise RuntimeError("无法读取首帧")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.AKAZE_create()
    ref_keypoints, ref_desc = detector.detectAndCompute(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    if ref_desc is None:
        raise RuntimeError("首帧特征提取失败")
    ref_points = [kp.pt for kp in ref_keypoints]
    # 参考框取整幅画面，跟踪时它的形变就是相机的整体运动。
    ref_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)

    writer = cv2.VideoWriter(os.path.join(out_dir, "akaze_tracking.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise IOError("无法创建输出视频")

    # 步骤 2~4：逐帧匹配、估计变换、绘制轨迹。
    samples = []
    index = 0
    tracked = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        canvas = frame.copy()
        quad = track_quad(detector, ref_points, ref_desc, ref_corners,
                          cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.ratio)
        if quad is not None:
            tracked += 1
            cv2.polylines(canvas, [np.int32(quad)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "frame {}".format(index), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        if index % 25 == 0:
            samples.append(canvas.copy())
        if args.display:
            cv2.imshow("akaze tracking", canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        index += 1

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print("输入: {}, 处理 {} 帧, 成功跟踪 {} 帧 ({:.1f}%)".format(
        input_path, index, tracked, 100.0 * tracked / index if index else 0.0))
    print("  参考帧关键点: {}".format(len(ref_keypoints)))
    print("  跟踪视频: {}".format(os.path.join(out_dir, "akaze_tracking.mp4")))

    panels = [fit_panel(samples[i], "frame {}".format(i * 25))
              for i in range(min(4, len(samples)))]
    if not panels:
        panels = [fit_panel(reference, "reference (no frame read)")]
    result = montage(panels, 2)
    output_path = os.path.join(out_dir, "akaze_tracking.png")
    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("  采样帧拼接: {}".format(os.path.abspath(output_path)))


if __name__ == "__main__":
    main()
