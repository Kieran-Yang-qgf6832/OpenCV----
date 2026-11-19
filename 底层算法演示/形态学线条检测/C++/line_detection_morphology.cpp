/**
 * @file    line_detection_morphology.cpp
 * @brief   形态学线条检测演示：用方向结构元素提取表格的水平线与垂直线。
 * @details 流程：二值化 -> 创建方向结构元素 -> 开运算提取线条 -> 轮廓筛选。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          核心思想：长度 1×N 的结构元素只能「容纳」水平方向的线条，开运算后其余结构被腐蚀掉；
 *          同理 1×M 的竖条只保留垂直线。结构元素长度要大于线宽、小于线长。
 */

#include <algorithm>
#include <iomanip>
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
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }
    const double scale =
        std::min(kPanelWidth / static_cast<double>(bgr.cols),
                 kPanelHeight / static_cast<double>(bgr.rows));
    cv::Mat scaled;
    cv::resize(bgr, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
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
 * @brief  过滤掉过短的线段，只保留真正的表格线。
 * @param  lines  候选线条二值图。
 * @param  minLen 最小长度。
 * @param  horizontal 是否为水平方向（决定用宽还是高做判据）。
 * @return 过滤后的二值图。
 */
cv::Mat filterByLength(const cv::Mat& lines, int minLen, bool horizontal)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(lines, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filtered = cv::Mat::zeros(lines.size(), CV_8U);
    for (const auto& contour : contours) {
        const cv::Rect box = cv::boundingRect(contour);
        const int length = horizontal ? box.width : box.height;
        if (length >= minLen) {
            cv::rectangle(filtered, box, cv::Scalar(255), cv::FILLED);
        }
    }
    return filtered;
}

}  // namespace

/**
 * @brief  主流程：提取水平线与垂直线并叠加显示。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/table.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/line_detection_morphology.png }"
        "{l lineLen      | 0    | 结构元素长度，0 表示自动取宽(高)的 1/25 }"
        "{n minLen       | 0    | 线条最小长度，0 表示自动取宽(高)的 1/20 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("形态学线条检测演示: 方向结构元素 + 开运算\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string inputPath = parser.get<std::string>("input");
    if (inputPath.empty()) {
        inputPath = locateAsset("table.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：二值化。表格线通常是深色，取反后成为前景。
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          15, -2);
    cv::Mat cleaned;
    cv::morphologyEx(binary, cleaned, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // 步骤 2：创建方向结构元素。长度取画面尺寸的 1/25 左右即可滤掉文字。
    int lineLen = parser.get<int>("lineLen");
    if (lineLen <= 0) {
        lineLen = std::max(10, std::min(gray.cols, gray.rows) / 25);
    }
    const cv::Mat horizontalElement =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(lineLen, 1));
    const cv::Mat verticalElement =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, lineLen));

    // 步骤 3：开运算提取线条。
    cv::Mat horizontal;
    cv::Mat vertical;
    cv::morphologyEx(cleaned, horizontal, cv::MORPH_OPEN, horizontalElement);
    cv::morphologyEx(cleaned, vertical, cv::MORPH_OPEN, verticalElement);

    // 步骤 4：轮廓筛选，去掉过短的残留。
    int minLen = parser.get<int>("minLen");
    if (minLen <= 0) {
        minLen = std::max(20, std::min(gray.cols, gray.rows) / 20);
    }
    horizontal = filterByLength(horizontal, minLen, true);
    vertical = filterByLength(vertical, minLen, false);

    cv::Mat horizontalLabels;
    cv::Mat verticalLabels;
    const int horizontalCount = cv::connectedComponents(horizontal, horizontalLabels) - 1;
    const int verticalCount = cv::connectedComponents(vertical, verticalLabels) - 1;
    std::cout << "输入: " << inputPath << ", 结构元素长度: " << lineLen
              << ", 最小线长: " << minLen << "\n";
    std::cout << "  水平线段: " << horizontalCount << ", 垂直线段: " << verticalCount << "\n";

    // 叠加显示：水平线绿色，垂直线红色。
    cv::Mat overlay = src.clone();
    overlay.setTo(cv::Scalar(0, 255, 0), horizontal);
    overlay.setTo(cv::Scalar(0, 0, 255), vertical);

    cv::Mat combined;
    cv::bitwise_or(horizontal, vertical, combined);

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(cleaned, "binary"),
        fitPanel(horizontal, "horizontal"),
        fitPanel(vertical, "vertical"),
        fitPanel(combined, "combined"),
        fitPanel(overlay, "overlay on original")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/line_detection_morphology.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("line detection", result);
        cv::waitKey(0);
    }
    return 0;
}
