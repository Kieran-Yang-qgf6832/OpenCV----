/**
 * @file    point_polygon_test.cpp
 * @brief   点多边形测试演示：判断像素点位于多边形内、外还是边界上，并给出带符号距离。
 * @details 流程：提取多边形轮廓 -> 网格布点 -> pointPolygonTest 计算距离 -> 分类着色
 *          -> 生成距离场可视化。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          measureDist=false 时函数只返回 +1/0/-1，速度快；true 时返回带符号的欧氏距离，
 *          内部为正、外部为负、边界为 0，是判断「离边界多远」的关键。
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

const int kPanelWidth = 480;
const int kPanelHeight = 360;
const int kTitleHeight = 30;

/** @brief 在若干候选位置中查找素材，便于从不同工作目录运行。 */
std::string locateAsset(const std::string& fileName)
{
    const std::vector<std::string> candidates = {
        "../../assets/" + fileName, "../assets/" + fileName, "assets/" + fileName};
    for (const auto& path : candidates) {
        if (cv::utils::fs::exists(path)) {
            return path;
        }
    }
    return candidates.front();
}

/** @brief 由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。 */
std::string outputDirFor(const std::string& inputPath)
{
    std::string dir = inputPath;
    const size_t slash = dir.find_last_of("/\\");
    dir = (slash == std::string::npos) ? "." : dir.substr(0, slash);
    const size_t slash2 = dir.find_last_of("/\\");
    const std::string leaf = (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
    if (leaf == "assets") {
        dir = (slash2 == std::string::npos) ? "." : dir.substr(0, slash2);
    }
    return dir + "/output";
}

/** @brief 等比缩放并居中留白到固定面板。 */
cv::Mat fitPanel(const cv::Mat& image, const std::string& title)
{
    const double scale =
        std::min(kPanelWidth / static_cast<double>(image.cols),
                 kPanelHeight / static_cast<double>(image.rows));
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::Mat panel(kPanelHeight + kTitleHeight, kPanelWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    const int x = (kPanelWidth - scaled.cols) / 2;
    const int y = kTitleHeight + (kPanelHeight - scaled.rows) / 2;
    scaled.copyTo(panel(cv::Rect(x, y, scaled.cols, scaled.rows)));
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/** @brief 按列数拼接等尺寸面板。 */
cv::Mat montage(const std::vector<cv::Mat>& panels, int cols)
{
    const int rows = static_cast<int>((panels.size() + cols - 1) / cols);
    cv::Mat canvas(rows * panels[0].rows, cols * panels[0].cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < panels.size(); ++i) {
        const int r = static_cast<int>(i) / cols;
        const int c = static_cast<int>(i) % cols;
        panels[i].copyTo(canvas(cv::Rect(c * panels[i].cols, r * panels[i].rows,
                                         panels[i].cols, panels[i].rows)));
    }
    return canvas;
}

/**
 * @brief  从二值图提取面积最大的外轮廓。
 * @param  binary 二值图。
 * @return 最大轮廓的点集。
 */
std::vector<cv::Point> largestContour(const cv::Mat& binary)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }
    const auto it = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    return *it;
}

/**
 * @brief  把多边形的带符号距离场量化为彩色图。
 * @param  contour 多边形。
 * @param  size    画布尺寸。
 * @param  step    采样步长（越小越精细也越慢）。
 * @return BGR 距离场可视化。
 */
cv::Mat distanceField(const std::vector<cv::Point>& contour, cv::Size size, int step)
{
    cv::Mat field(size, CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < size.height; y += step) {
        for (int x = 0; x < size.width; x += step) {
            const double distance =
                cv::pointPolygonTest(contour, cv::Point2f(static_cast<float>(x),
                                                         static_cast<float>(y)), true);
            const int w = std::min(step, size.width - x);
            const int h = std::min(step, size.height - y);
            field(cv::Rect(x, y, w, h)).setTo(distance);
        }
    }
    cv::Mat normalized;
    cv::normalize(field, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat normalized8;
    normalized.convertTo(normalized8, CV_8U);  // applyColorMap 只接受 CV_8UC1 / CV_8UC3
    cv::Mat colored;
    cv::applyColorMap(normalized8, colored, cv::COLORMAP_JET);
    return colored;
}

}  // namespace

/**
 * @brief  主流程：提取多边形、网格布点分类并输出距离场。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{b binary       |      | 二值图，缺省读取 assets/binary.png }"
        "{s shapes       |      | 用于叠加可视化的彩色图，缺省读取 assets/shapes.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/point_polygon_test.png }"
        "{g grid         | 40   | 测试点的网格间距(像素) }"
        "{p step         | 6    | 距离场的采样步长(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("点多边形测试演示: pointPolygonTest 的位置关系与距离\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string binaryPath = parser.get<std::string>("binary");
    std::string shapesPath = parser.get<std::string>("shapes");
    if (binaryPath.empty()) {
        binaryPath = locateAsset("binary.png");
    }
    if (shapesPath.empty()) {
        shapesPath = locateAsset("shapes.png");
    }

    // 步骤 1：定义多边形。二值图转灰度后提取最大外轮廓。
    const cv::Mat binarySrc = cv::imread(binaryPath, cv::IMREAD_GRAYSCALE);
    cv::Mat canvas = cv::imread(shapesPath, cv::IMREAD_COLOR);
    if (binarySrc.empty() || canvas.empty()) {
        std::cerr << "无法读取素材" << std::endl;
        return -1;
    }
    cv::Mat binary;
    // 素材是白底黑形状：必须取反，否则 findContours 会把整幅画面的白色边框当成最大轮廓。
    cv::threshold(binarySrc, binary, 127, 255, cv::THRESH_BINARY_INV);
    const std::vector<cv::Point> polygon = largestContour(binary);
    if (polygon.size() < 3) {
        std::cerr << "未找到有效多边形轮廓" << std::endl;
        return -1;
    }

    // 步骤 2：测试多个点，并用 measureDist=true 拿到带符号距离。
    const int grid = std::max(4, parser.get<int>("grid"));
    int inside = 0;
    int outside = 0;
    int onEdge = 0;
    double maxInside = 0.0;
    double minOutside = 0.0;
    const cv::Scalar kInsideColor(60, 200, 60);
    const cv::Scalar kOutsideColor(60, 60, 220);
    const cv::Scalar kEdgeColor(220, 180, 40);

    cv::Mat classified = canvas.clone();
    for (int y = grid / 2; y < canvas.rows; y += grid) {
        for (int x = grid / 2; x < canvas.cols; x += grid) {
            const cv::Point2f pt(static_cast<float>(x), static_cast<float>(y));
            const double distance = cv::pointPolygonTest(polygon, pt, true);
            cv::Scalar color = kOutsideColor;
            if (distance > 0.0) {
                ++inside;
                color = kInsideColor;
                maxInside = std::max(maxInside, distance);
            } else if (distance < 0.0) {
                ++outside;
                minOutside = std::min(minOutside, distance);
            } else {
                ++onEdge;
                color = kEdgeColor;
            }
            // 半径随距离增大，直观看出「离边界多远」。
            const int radius = std::max(2, std::min(9, 2 + static_cast<int>(std::fabs(distance) / 25.0)));
            cv::circle(classified, cv::Point(x, y), radius, color, cv::FILLED, cv::LINE_AA);
        }
    }
    cv::drawContours(classified, std::vector<std::vector<cv::Point>>{polygon}, -1,
                     cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    // 步骤 3：可视化。
    cv::Mat contourView;
    cv::cvtColor(binary, contourView, cv::COLOR_GRAY2BGR);
    cv::drawContours(contourView, std::vector<std::vector<cv::Point>>{polygon}, -1,
                     cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    const int step = std::max(2, parser.get<int>("step"));
    const cv::Mat field = distanceField(polygon, canvas.size(), step);

    std::cout << "多边形顶点数: " << polygon.size() << ", 面积: " << cv::contourArea(polygon)
              << "\n网格间距: " << grid << " 像素\n";
    std::cout << "  内部点: " << inside << ", 外部点: " << outside << ", 边界点: " << onEdge << "\n";
    std::cout << "  最大内部距离(内切半径): " << maxInside << "\n";
    std::cout << "  外部点最近距离: " << minOutside << "\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(contourView, "largest contour"),
        fitPanel(classified, "inside / outside / edge"),
        fitPanel(field, "signed distance field")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(binaryPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/point_polygon_test.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("point polygon test", result);
        cv::waitKey(0);
    }
    return 0;
}
