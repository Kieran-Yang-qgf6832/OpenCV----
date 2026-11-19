# -*- coding: utf-8 -*-
"""
@file    planar_object_detection.py
@brief   平面物体检测演示：特征匹配 + 单应估计 + 姿态分解，在场景中定位平面物体。
@details 流程：特征检测匹配 -> 计算单应性 -> 分解姿态 -> 绘制检测框。
@note    依赖：opencv-python、numpy
         decomposeHomographyMat 需要相机内参；这里没有标定数据，采用「焦距 ≈ 1.2×长边、
         主点在画面中心」的虚拟内参，因此分解出的平移量只有相对意义，旋转较可靠。
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


def main():
    """@brief 主流程：匹配、估计 H、分解姿态并绘制检测框。"""
    parser = argparse.ArgumentParser(
        description="平面物体检测演示：特征匹配 + 单应估计 + 姿态分解")
    parser.add_argument("-s", "--scene", default=None,
                        help="场景图，缺省读取 assets/scene.png")
    parser.add_argument("-t", "--template", default=None,
                        help="模板图，缺省读取 assets/template.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/planar_object_detection.png")
    parser.add_argument("-n", "--features", type=int, default=2000,
                        help="ORB 特征点数上限")
    parser.add_argument("-r", "--ratio", type=float, default=0.75, help="Lowe 比率阈值")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    scene_path = args.scene or locate_asset("scene.png")
    template_path = args.template or locate_asset("template.png")

    scene_image = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    template_image = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if scene_image is None or template_image is None:
        raise FileNotFoundError("无法读取图像")

    # 步骤 1：特征检测匹配。
    orb = cv2.ORB_create(max(100, args.features))
    kp_template, desc_template = orb.detectAndCompute(template_image, None)
    kp_scene, desc_scene = orb.detectAndCompute(scene_image, None)
    if desc_template is None or desc_scene is None:
        raise RuntimeError("特征提取失败")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_template, desc_scene, k=2)
    good = [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < args.ratio * pair[1].distance]
    if len(good) < 4:
        raise RuntimeError("有效匹配不足 4 组")

    pts_template = np.float32([kp_template[m.queryIdx].pt for m in good])
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good])

    # 步骤 2：计算单应性。RANSAC 剔除错误匹配。
    homography, inlier_mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
    if homography is None:
        raise RuntimeError("单应估计失败")
    inliers = int(inlier_mask.sum())

    # 步骤 3：分解姿态。把模板的四个角变换到场景中，顺便得到检测框。
    h, w = template_image.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)

    # 虚拟内参：无标定数据时的常用近似，平移量只有相对意义。
    scene_h, scene_w = scene_image.shape[:2]
    focal = 1.2 * max(scene_w, scene_h)
    camera_matrix = np.array([[focal, 0, scene_w / 2.0],
                              [0, focal, scene_h / 2.0],
                              [0, 0, 1]], dtype=np.float64)
    try:
        solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
            homography, camera_matrix)
    except cv2.error:
        solutions, rotations, translations = 0, [], []

    print("匹配数: {}, 内点: {} ({:.1f}%)".format(len(good), inliers, 100.0 * inliers / len(good)))
    print("姿态分解解数: {}".format(solutions))
    if rotations:
        rvec, _ = cv2.Rodrigues(rotations[0])
        print("  第 1 组解 旋转向量: [{:.4f}, {:.4f}, {:.4f}] rad".format(*rvec.ravel()))
        print("  第 1 组解 平移向量: [{:.4f}, {:.4f}, {:.4f}]".format(*translations[0].ravel()))
        print("  注: 使用虚拟内参(f={:.1f})，平移量仅有相对意义".format(focal))

    # 步骤 4：绘制检测框。
    scene_view = scene_image.copy()
    polygon = projected.astype(np.int32)
    cv2.polylines(scene_view, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
    for pt in polygon:
        cv2.circle(scene_view, tuple(pt), 6, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

    # 匹配可视化与矫正结果。
    matches_view = cv2.drawMatches(
        template_image, kp_template, scene_image, kp_scene, good, None,
        matchesMask=inlier_mask.ravel().astype(int).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    rectified = cv2.warpPerspective(scene_image, np.linalg.inv(homography), (w, h))

    panels = [
        fit_panel(template_image, "template"),
        fit_panel(scene_view, "detected: {} inliers".format(inliers)),
        fit_panel(matches_view, "inlier matches"),
        fit_panel(rectified, "rectified object"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(scene_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "planar_object_detection.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("planar object detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
