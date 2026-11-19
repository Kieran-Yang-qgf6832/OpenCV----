/**
 * @file    color_segmentation.cpp
 * @brief   颜色分割演示：在 HSV 空间按色相区间提取目标颜色区域。
 * @details 流程：转换 HSV -> 设定颜色阈值 -> 生成掩码提取目标 -> 形态学优化掩码。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          HSV 的色相是环形的，红色跨 0° 附近，必须用两段区间（0~10 与 170~179）合并；
 *          直接在 BGR 上设阈值很难把「苹果的红」与「阴影的暗红」分开。
 */

#include <algorithm>
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
 * @brief  主流程：HSV 色相区间分割 + 形态学优化。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/apples.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/color_segmentation.png }"
        "{n minSaturation| 90   | 最小饱和度，滤掉发白区域 }"
        "{v minValue     | 60   | 最小明度，滤掉过暗区域 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("颜色分割演示: HSV 色相阈值 + inRange + 形态学\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("apples.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：转换到 HSV。
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);

    // 步骤 2：设定颜色阈值。红色跨色相 0 点，用两段区间合并。
    const int minS = parser.get<int>("minSaturation");
    const int minV = parser.get<int>("minValue");
    cv::Mat maskLow;
    cv::Mat maskHigh;
    cv::inRange(hsv, cv::Scalar(0, minS, minV), cv::Scalar(10, 255, 255), maskLow);
    cv::inRange(hsv, cv::Scalar(170, minS, minV), cv::Scalar(179, 255, 255), maskHigh);

    // 步骤 3：生成掩码并提取目标。
    cv::Mat mask;
    cv::bitwise_or(maskLow, maskHigh, mask);
    cv::Mat extracted;
    src.copyTo(extracted, mask);

    // 步骤 4：形态学优化掩码（开运算去噪、闭运算填洞）。
    cv::Mat refined;
    const cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, refined, cv::MORPH_OPEN, element);
    cv::morphologyEx(refined, refined, cv::MORPH_CLOSE, element);
    cv::Mat refinedExtracted;
    src.copyTo(refinedExtracted, refined);

    const double rawRatio = 100.0 * cv::countNonZero(mask) / mask.total();
    const double refinedRatio = 100.0 * cv::countNonZero(refined) / refined.total();
    std::cout << "输入: " << inputPath << ", S>=" << minS << ", V>=" << minV << "\n";
    std::cout << "  原始掩码占比: " << rawRatio << "%\n";
    std::cout << "  形态学优化后: " << refinedRatio << "%\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(src, "original"),
        fitPanel(hsvChannels[0], "hue channel"),
        fitPanel(mask, "raw mask"),
        fitPanel(refined, "refined mask"),
        fitPanel(extracted, "extracted (raw)"),
        fitPanel(refinedExtracted, "extracted (refined)")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/color_segmentation.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("color segmentation", result);
        cv::waitKey(0);
    }
    return 0;
}
