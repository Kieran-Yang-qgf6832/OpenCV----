/**
 * @file    color_space_conversion.cpp
 * @brief   色彩空间转换演示：在 BGR / HSV / Lab / YCrCb 之间转换并分析各通道特性。
 * @details 流程：读取彩色图 -> 转换目标空间 -> 分离通道并可视化 -> 统计各通道特性
 *          -> 用 HSV 色相区间做颜色分割，说明换空间的实用价值。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          HSV 的 H 取值 0~179（8 位图角度折半），Lab 的 a/b 以 128 为零点，
 *          直接当灰度显示会很暗或很平，展示时需归一化。
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

/// 拼接图中标题条的高度（像素）。
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

/**
 * @brief  把单通道图归一化到 0~255 并加标题条，转成三通道便于拼接。
 * @param  channel 单通道图像。
 * @param  title   标题文字（仅支持 ASCII）。
 * @return 带标题条的三通道面板。
 * @note   Lab 的 a/b 通道以 128 为零点、动态范围窄，不归一化几乎看不见差异。
 */
cv::Mat channelPanel(const cv::Mat& channel, const std::string& title)
{
    // 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    cv::Mat small;
    cv::resize(channel, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat normalized;
    cv::normalize(small, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::cvtColor(normalized, panel(cv::Rect(0, kTitleHeight, small.cols, small.rows)),
                 cv::COLOR_GRAY2BGR);
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/**
 * @brief  给彩色图加标题条（不做归一化）。
 * @param  image 三通道图像。
 * @param  title 标题文字。
 * @return 带标题条的面板。
 */
cv::Mat colorPanel(const cv::Mat& image, const std::string& title)
{
    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    small.copyTo(panel(cv::Rect(0, kTitleHeight, small.cols, small.rows)));
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/**
 * @brief  按列数拼接面板。
 * @param  panels 面板列表。
 * @param  cols   每行面板数。
 * @return 拼接结果。
 */
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
 * @brief  打印某个空间的各通道统计量。
 * @param  spaceName 空间名称。
 * @param  channels  通道列表。
 * @param  names     通道名称列表。
 */
void printChannelStats(const std::string& spaceName,
                       const std::vector<cv::Mat>& channels,
                       const std::vector<std::string>& names)
{
    std::cout << spaceName << ":\n";
    for (size_t i = 0; i < channels.size(); ++i) {
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(channels[i], mean, stddev);
        std::cout << "  " << std::left << std::setw(4) << names[i]
                  << " mean=" << std::fixed << std::setprecision(1) << std::setw(6) << mean[0]
                  << " std=" << std::setw(6) << stddev[0] << "\n";
    }
}

}  // namespace

/**
 * @brief  主流程：转换三种色彩空间、分离通道、统计特性并做 HSV 颜色分割。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/fruits.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/color_space_conversion.png }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("色彩空间转换演示: BGR / HSV / Lab / YCrCb 通道分析与 HSV 分割\n"
                 "参数请使用 --key=value 形式");
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
        inputPath = locateAsset("fruits.png");
    }

    // 步骤 1：读取图像。
    const cv::Mat bgr = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 2：转换到目标色彩空间。
    cv::Mat hsv;
    cv::Mat lab;
    cv::Mat ycrcb;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);

    // 步骤 3：分离通道。
    std::vector<cv::Mat> hsvChannels;
    std::vector<cv::Mat> labChannels;
    std::vector<cv::Mat> ycrcbChannels;
    cv::split(hsv, hsvChannels);
    cv::split(lab, labChannels);
    cv::split(ycrcb, ycrcbChannels);

    // 步骤 4：分析特性（均值反映整体色调，标准差反映通道内的分布范围）。
    printChannelStats("HSV (H:0~179, S/V:0~255)", hsvChannels, {"H", "S", "V"});
    printChannelStats("Lab (L:0~255, a/b:以128为零点)", labChannels, {"L", "a", "b"});
    printChannelStats("YCrCb", ycrcbChannels, {"Y", "Cr", "Cb"});

    // 步骤 5：用 HSV 的色相区间做颜色分割——这是换色彩空间最实用的场景。
    //        柠檬的黄色在 OpenCV 的 H 轴上大约落在 20~35。
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(20, 80, 80), cv::Scalar(35, 255, 255), mask);
    cv::Mat masked;
    bgr.copyTo(masked, mask);
    std::cout << "黄色像素占比: " << std::fixed << std::setprecision(2)
              << 100.0 * cv::countNonZero(mask) / mask.total() << "%\n";

    // 拼接结果。
    const std::vector<cv::Mat> panels = {
        colorPanel(bgr, "BGR original"),
        channelPanel(hsvChannels[0], "HSV H (hue)"),
        channelPanel(hsvChannels[1], "HSV S (saturation)"),
        channelPanel(hsvChannels[2], "HSV V (value)"),
        channelPanel(labChannels[0], "Lab L (lightness)"),
        channelPanel(labChannels[1], "Lab a (green-red)"),
        channelPanel(labChannels[2], "Lab b (blue-yellow)"),
        channelPanel(ycrcbChannels[0], "YCrCb Y (luma)"),
        channelPanel(mask, "H in [20,35] mask"),
        colorPanel(masked, "yellow only")};
    const cv::Mat result = montage(panels, 5);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/color_space_conversion.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("color space conversion", result);
        cv::waitKey(0);
    }
    return 0;
}
