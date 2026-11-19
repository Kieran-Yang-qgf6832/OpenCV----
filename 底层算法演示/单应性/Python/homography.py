# -*- coding: utf-8 -*-
"""
@file    homography.py
@brief   单应性演示：特征匹配 + RANSAC 估计 3x3 单应矩阵，用于透视矫正与图像拼接。
@details 流程：特征匹配(ORB) -> RANSAC 估计 H -> 透视变换(矫正) -> 图像拼接 -> 输出对比图。
@note    依赖：opencv-python、numpy
         单应矩阵只在「同一平面」或「相机纯旋转」时成立；有视差的三维场景应改用基础矩阵。
"""

import argparse
import os

import cv2
import numpy as np

# 拼接图里每个面板的尺寸（长边等比缩放后居中留白）。
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
    """
    @brief  把图像等比缩放到固定面板内并居中留白，保证拼接时各面板尺寸一致。
    @param  image 输入图像（任意尺寸）。
    @param  title 标题文字（仅支持 ASCII）。
    @return 固定尺寸的面板。
    """
    scale = min(PANEL_WIDTH / image.shape[1], PANEL_HEIGHT / image.shape[0])
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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


def match_features(ref_image, scene_image, ratio=0.75):
    """
    @brief  用 ORB 做特征匹配，并用 Lowe 比率筛选。
    @param  ref_image   参考图（模板）。
    @param  scene_image 目标图（场景）。
    @param  ratio       比率阈值，0.75 是常用值。
    @return (kp_ref, kp_scene, matches)
    """
    orb = cv2.ORB_create(2000)
    kp_ref, desc_ref = orb.detectAndCompute(ref_image, None)
    kp_scene, desc_scene = orb.detectAndCompute(scene_image, None)
    if desc_ref is None or desc_scene is None:
        return kp_ref, kp_scene, []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_ref, desc_scene, k=2)
    # 最近邻明显优于次近邻才保留，可过滤掉大量错误匹配。
    matches = [pair[0] for pair in knn
               if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    return kp_ref, kp_scene, matches


def print_homography(h_matrix):
    """
    @brief  打印归一化后的 3x3 单应矩阵。
    @param  h_matrix 单应矩阵。
    @note   单应矩阵只在相差一个尺度因子的意义下唯一，统一把 h33 归一化为 1 才便于比较。
    """
    scale = h_matrix[2, 2] if h_matrix[2, 2] != 0 else 1.0
    print("H (归一化 h33=1):")
    for row in h_matrix / scale:
        print("  [" + "".join("{:12.6f}".format(v) for v in row) + " ]")


def main():
    """@brief 主流程：匹配、估计 H、透视矫正与拼接。"""
    parser = argparse.ArgumentParser(
        description="单应性演示：ORB 匹配 + RANSAC + 透视矫正/拼接")
    parser.add_argument("-r", "--ref", default=None,
                        help="参考图，缺省读取 assets/poster.png")
    parser.add_argument("-s", "--scene", default=None,
                        help="场景图，缺省读取 assets/poster_perspective.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/homography.png")
    parser.add_argument("-c", "--ratio", type=float, default=0.75, help="Lowe 比率阈值")
    parser.add_argument("-t", "--thresh", type=float, default=3.0,
                        help="RANSAC 重投影误差阈值(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    ref_path = args.ref or locate_asset("poster.png")
    scene_path = args.scene or locate_asset("poster_perspective.png")

    ref_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    scene_image = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    if ref_image is None or scene_image is None:
        missing = ref_path if ref_image is None else scene_path
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(missing)))

    # 步骤 1：特征匹配。
    kp_ref, kp_scene, matches = match_features(ref_image, scene_image, args.ratio)
    if len(matches) < 4:
        raise RuntimeError("匹配点不足 4 组，无法求解单应矩阵（当前 {}）".format(len(matches)))

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches])

    # 步骤 2：RANSAC 估计单应矩阵，mask 标记内点。
    h_matrix, inlier_mask = cv2.findHomography(pts_ref, pts_scene, cv2.RANSAC, args.thresh)
    if h_matrix is None:
        raise RuntimeError("RANSAC 求解失败")
    inliers = int(inlier_mask.sum())
    print("匹配数: {}, RANSAC 内点: {} ({:.1f}%)".format(
        len(matches), inliers, 100.0 * inliers / len(matches)))
    print_homography(h_matrix)

    # 步骤 3：透视变换。H 是 ref -> scene，矫正场景图要用它的逆矩阵。
    h_inv = np.linalg.inv(h_matrix)
    rectified = cv2.warpPerspective(
        scene_image, h_inv, (ref_image.shape[1], ref_image.shape[0]))

    # 步骤 4：图像拼接。把参考图按 H 投到场景图上，只在有效区域内做 50% 混合。
    warped = cv2.warpPerspective(
        ref_image, h_matrix, (scene_image.shape[1], scene_image.shape[0]))
    _, warped_mask = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
                                   1, 255, cv2.THRESH_BINARY)
    blended = scene_image.copy()
    blended_roi = cv2.addWeighted(scene_image, 0.5, warped, 0.5, 0.0)
    blended[warped_mask > 0] = blended_roi[warped_mask > 0]

    # 对比图：全部匹配 vs 仅内点。
    all_match_image = cv2.drawMatches(
        ref_image, kp_ref, scene_image, kp_scene, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    inlier_image = cv2.drawMatches(
        ref_image, kp_ref, scene_image, kp_scene, matches, None,
        matchesMask=inlier_mask.ravel().astype(int).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    panels = [
        fit_panel(ref_image, "reference (planar)"),
        fit_panel(scene_image, "scene (with perspective)"),
        fit_panel(all_match_image, "matches: {}".format(len(matches))),
        fit_panel(inlier_image, "inliers: {}".format(inliers)),
        fit_panel(rectified, "rectified by inv(H)"),
        fit_panel(blended, "stitched"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(ref_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "homography.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("homography", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
