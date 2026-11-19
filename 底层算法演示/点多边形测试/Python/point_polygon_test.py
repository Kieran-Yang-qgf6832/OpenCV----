# -*- coding: utf-8 -*-
"""
@file    point_polygon_test.py
@brief   点多边形测试演示：判断像素点位于多边形内、外还是边界上，并给出带符号距离。
@details 流程：提取多边形轮廓 -> 网格布点 -> pointPolygonTest 计算距离 -> 分类着色
         -> 生成距离场可视化。
@note    依赖：opencv-python、numpy
         measureDist=False 时函数只返回 +1/0/-1，速度快；True 时返回带符号的欧氏距离，
         内部为正、外部为负、边界为 0，是判断「离边界多远」的关键。
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


def largest_contour(binary):
    """
    @brief  从二值图提取面积最大的外轮廓。
    @param  binary 二值图。
    @return 最大轮廓的点集（Nx1x2）。
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def distance_field(polygon, size, step):
    """
    @brief  把多边形的带符号距离场量化为彩色图。
    @param  polygon 多边形轮廓。
    @param  size    画布尺寸 (width, height)。
    @param  step    采样步长（越小越精细也越慢）。
    @return BGR 距离场可视化。
    """
    width, height = size
    field = np.zeros((height, width), dtype=np.float32)
    for y in range(0, height, step):
        for x in range(0, width, step):
            distance = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
            field[y:y + step, x:x + step] = distance
    normalized = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def main():
    """@brief 主流程：提取多边形、网格布点分类并输出距离场。"""
    parser = argparse.ArgumentParser(
        description="点多边形测试演示：pointPolygonTest 的位置关系与距离")
    parser.add_argument("-b", "--binary", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-s", "--shapes", default=None,
                        help="用于叠加可视化的彩色图，缺省读取 assets/shapes.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/point_polygon_test.png")
    parser.add_argument("-g", "--grid", type=int, default=40, help="测试点的网格间距(像素)")
    parser.add_argument("-p", "--step", type=int, default=6, help="距离场的采样步长(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    binary_path = args.binary or locate_asset("binary.png")
    shapes_path = args.shapes or locate_asset("shapes.png")

    # 步骤 1：定义多边形。二值图转灰度后提取最大外轮廓。
    binary_src = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    canvas = cv2.imread(shapes_path, cv2.IMREAD_COLOR)
    if binary_src is None or canvas is None:
        raise FileNotFoundError("无法读取素材")
    # 素材是白底黑形状：必须取反，否则 findContours 会把整幅画面的白色边框当成最大轮廓。
    _, binary = cv2.threshold(binary_src, 127, 255, cv2.THRESH_BINARY_INV)
    polygon = largest_contour(binary)
    if polygon is None or len(polygon) < 3:
        raise RuntimeError("未找到有效多边形轮廓")

    # 步骤 2：测试多个点，并用 measureDist=True 拿到带符号距离。
    grid = max(4, args.grid)
    classified = canvas.copy()
    stats = {"inside": 0, "outside": 0, "edge": 0}
    max_inside = 0.0
    min_outside = 0.0
    for y in range(grid // 2, canvas.shape[0], grid):
        for x in range(grid // 2, canvas.shape[1], grid):
            distance = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
            if distance > 0:
                stats["inside"] += 1
                color = (60, 200, 60)
                max_inside = max(max_inside, distance)
            elif distance < 0:
                stats["outside"] += 1
                color = (60, 60, 220)
                min_outside = min(min_outside, distance)
            else:
                stats["edge"] += 1
                color = (220, 180, 40)
            # 半径随距离增大，直观看出「离边界多远」。
            radius = max(2, min(9, 2 + int(abs(distance) / 25.0)))
            cv2.circle(classified, (x, y), radius, color, cv2.FILLED, cv2.LINE_AA)
    cv2.drawContours(classified, [polygon], -1, (0, 0, 0), 2, cv2.LINE_AA)

    # 步骤 3：可视化。
    contour_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_view, [polygon], -1, (0, 0, 255), 2, cv2.LINE_AA)
    field = distance_field(polygon, (canvas.shape[1], canvas.shape[0]), max(2, args.step))

    print("多边形顶点数: {}, 面积: {:.0f}".format(len(polygon), cv2.contourArea(polygon)))
    print("网格间距: {} 像素".format(grid))
    print("  内部点: {}, 外部点: {}, 边界点: {}".format(
        stats["inside"], stats["outside"], stats["edge"]))
    print("  最大内部距离(内切半径): {:.2f}".format(max_inside))
    print("  外部点最近距离: {:.2f}".format(min_outside))

    panels = [
        fit_panel(contour_view, "largest contour"),
        fit_panel(classified, "inside / outside / edge"),
        fit_panel(field, "signed distance field"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(binary_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "point_polygon_test.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("point polygon test", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
