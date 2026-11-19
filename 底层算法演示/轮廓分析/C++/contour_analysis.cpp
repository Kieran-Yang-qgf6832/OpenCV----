/**
 * @file    contour_analysis.cpp
 * @brief   轮廓分析演示：查找轮廓并计算面积、周长、圆度、多边形近似与包围盒。
 * @details 流程：二值化 -> 查找轮廓 -> 遍历计算属性 -> 绘制轮廓与标注。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          轮廓的「点」没有面积语义上的顺序保证，面积用 contourArea（格林公式），
 *          周长用 arcLength；圆度 4πA/P² 对完美圆为 1，可用于形状分类。
 */

#include <algorithm>
#include <cmath>
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

}  // namespace

/**
 * @brief  主流程：提取轮廓、计算属性并可视化。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/contour_analysis.png }"
        "{a minArea      | 500  | 最小面积过滤阈值 }"
        "{e epsilon      | 0.02 | 多边形近似的精度，占周长的比例 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("轮廓分析演示: 面积/周长/圆度/多边形近似\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("binary.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景。
    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY_INV);

    // 步骤 2：查找外轮廓。
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contourView;
    cv::cvtColor(binary, contourView, cv::COLOR_GRAY2BGR);
    cv::Mat approxView = contourView.clone();
    cv::Mat boxView = contourView.clone();
    const double epsilonRatio = parser.get<double>("epsilon");
    const int minArea = parser.get<int>("minArea");

    // 步骤 3：遍历计算属性。
    int kept = 0;
    std::cout << "输入: " << inputPath << ", 轮廓总数: " << contours.size() << "\n";
    std::cout << std::left << std::setw(4) << "id" << std::setw(10) << "area"
              << std::setw(10) << "perim" << std::setw(10) << "circular"
              << std::setw(10) << "vertices" << "bbox\n";
    cv::RNG rng(20261119);
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area < minArea) {
            continue;
        }
        ++kept;
        const double perimeter = cv::arcLength(contours[i], true);
        const double circularity =
            perimeter > 0.0 ? 4.0 * CV_PI * area / (perimeter * perimeter) : 0.0;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilonRatio * perimeter, true);
        const cv::Rect box = cv::boundingRect(contours[i]);

        const cv::Scalar color(rng.uniform(40, 255), rng.uniform(40, 255), rng.uniform(40, 255));
        cv::drawContours(contourView, contours, static_cast<int>(i), color, 2, cv::LINE_AA);
        cv::drawContours(approxView, std::vector<std::vector<cv::Point>>{approx}, -1,
                         cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        for (const auto& pt : approx) {
            cv::circle(approxView, pt, 4, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        }
        cv::rectangle(boxView, box, color, 2, cv::LINE_AA);
        cv::putText(boxView, "A=" + std::to_string(static_cast<int>(area)),
                    cv::Point(box.x + 2, box.y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv::LINE_AA);

        if (kept <= 12) {
            std::cout << std::left << std::setw(4) << i
                      << std::setw(10) << static_cast<int>(area)
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << perimeter
                      << std::setw(10) << circularity
                      << std::setw(10) << approx.size()
                      << box.x << "," << box.y << "," << box.width << "," << box.height << "\n";
        }
    }
    std::cout << "  面积 >= " << minArea << " 的轮廓: " << kept << " 个\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(binary, "binary input"),
        fitPanel(contourView, "contours: " + std::to_string(kept)),
        fitPanel(approxView, "approxPolyDP (eps " + std::to_string(epsilonRatio).substr(0, 4) + ")"),
        fitPanel(boxView, "bounding boxes")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/contour_analysis.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("contour analysis", result);
        cv::waitKey(0);
    }
    return 0;
}
