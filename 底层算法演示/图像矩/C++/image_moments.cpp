/**
 * @file    image_moments.cpp
 * @brief   图像矩演示：计算轮廓的几何矩、质心与 Hu 矩，并用 Hu 矩做形状匹配。
 * @details 流程：提取轮廓 -> 计算图像矩 -> 计算质心 -> 计算 Hu 矩 -> 形状匹配。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          Hu 矩有 7 个分量，对平移、缩放、旋转不变；数值跨越多阶数量级，
 *          比较前通常取 -sign(h)·log10|h|，本演示打印时直接给出 log 值。
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
 * @brief  主流程：逐轮廓计算矩特征并与第一个轮廓做形状匹配。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_moments.png }"
        "{a minArea      | 500  | 最小面积过滤阈值 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像矩演示: moments / centroid / HuMoments / matchShapes\n参数请使用 --key=value 形式");
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

    // 步骤 1：提取轮廓。素材白底黑形状，取反后形状成为前景。
    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat binaryView;
    cv::cvtColor(binary, binaryView, cv::COLOR_GRAY2BGR);
    cv::Mat momentView = binaryView.clone();

    const int minArea = parser.get<int>("minArea");
    std::vector<std::vector<cv::Point>> kept;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= minArea) {
            kept.push_back(contour);
        }
    }
    if (kept.empty()) {
        std::cerr << "没有面积达到阈值的轮廓" << std::endl;
        return -1;
    }

    std::cout << "输入: " << inputPath << ", 有效轮廓: " << kept.size() << "\n";
    std::cout << std::left << std::setw(4) << "id" << std::setw(10) << "area"
              << std::setw(20) << "centroid" << std::setw(10) << "hu[0]"
              << std::setw(10) << "hu[1]" << "match\n";

    // 步骤 2~4：图像矩、质心、Hu 矩、形状匹配。
    for (size_t i = 0; i < kept.size(); ++i) {
        const cv::Moments m = cv::moments(kept[i]);
        if (m.m00 == 0.0) {
            continue;
        }
        const double cx = m.m10 / m.m00;
        const double cy = m.m01 / m.m00;

        double hu[7];
        cv::HuMoments(m, hu);
        // Hu 矩跨数量级，取 -sign(h)·log10|h| 才便于比较与显示。
        const double hu0 = -std::copysign(1.0, hu[0]) * std::log10(std::fabs(hu[0]) + 1e-30);
        const double hu1 = -std::copysign(1.0, hu[1]) * std::log10(std::fabs(hu[1]) + 1e-30);

        // matchShapes 越小越相似；这里统一和第一个轮廓比较。
        const double match = cv::matchShapes(kept[i], kept[0], cv::CONTOURS_MATCH_I1, 0.0);

        std::cout << std::left << std::setw(4) << i << std::setw(10)
                  << static_cast<int>(cv::contourArea(kept[i])) << std::setw(20)
                  << (std::to_string(static_cast<int>(cx)) + ", " +
                      std::to_string(static_cast<int>(cy)))
                  << std::fixed << std::setprecision(2) << std::setw(10) << hu0
                  << std::setw(10) << hu1 << std::setprecision(4) << match << "\n";

        cv::drawContours(momentView, kept, static_cast<int>(i), cv::Scalar(0, 200, 0), 2,
                         cv::LINE_AA);
        cv::circle(momentView, cv::Point(cvRound(cx), cvRound(cy)), 6,
                   cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        char label[32];
        std::snprintf(label, sizeof(label), "#%zu", i);
        cv::putText(momentView, label, cv::Point(cvRound(cx) + 8, cvRound(cy) - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // 第三格：按 matchShapes 得分着色（越相似越偏绿）。
    cv::Mat matchView = binaryView.clone();
    for (size_t i = 0; i < kept.size(); ++i) {
        const double match = cv::matchShapes(kept[i], kept[0], cv::CONTOURS_MATCH_I1, 0.0);
        const int green = std::max(0, 255 - static_cast<int>(match * 120.0));
        cv::drawContours(matchView, kept, static_cast<int>(i),
                         cv::Scalar(0, green, 255 - green), cv::FILLED, cv::LINE_AA);
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(binary, "binary input"),
        fitPanel(momentView, "centroids & Hu"),
        fitPanel(matchView, "matchShapes score")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_moments.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image moments", result);
        cv::waitKey(0);
    }
    return 0;
}
