/**
 * @file    hit_or_miss.cpp
 * @brief   击中击不中演示：用形态学核精确匹配特定尺寸的孤立形状。
 * @details 流程：定义击中/击不中核 -> 应用变换 -> 定位匹配 -> 统计并标记。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          核的取值约定：1 表示该位置必须是前景，-1 表示必须是背景，0 表示不关心。
 *          素材是 5x5 块重复排布的图案（35x35 方块、17x17 方块、59x9 横杠各 25 个），
 *          因此可以用「尺寸 + 一圈背景」的核唯一锁定某一类形状。
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

/**
 * @brief  构造「精确尺寸 + 一圈背景」的击中击不中核。
 * @param  size 目标形状的边长（像素）。
 * @return CV_8S 核：内部 1（必须前景），外环 -1（必须背景）。
 */
cv::Mat makeIsolationKernel(int size)
{
    cv::Mat kernel(size + 2, size + 2, CV_8S, cv::Scalar(-1));
    kernel(cv::Rect(1, 1, size, size)).setTo(1);
    return kernel;
}

/**
 * @brief  把 CV_8S 核渲染成可看的灰度图（前景白、背景黑、不关心灰）。
 * @param  kernel 击中击不中核。
 * @param  scale  放大倍数。
 * @return 可视化图像。
 */
cv::Mat visualizeKernel(const cv::Mat& kernel, int scale)
{
    cv::Mat view8(kernel.size(), CV_8U);
    for (int r = 0; r < kernel.rows; ++r) {
        for (int c = 0; c < kernel.cols; ++c) {
            const schar v = kernel.at<schar>(r, c);
            view8.at<uchar>(r, c) = (v > 0) ? 255 : (v < 0 ? 0 : 128);
        }
    }
    cv::Mat big;
    cv::resize(view8, big, cv::Size(), scale, scale, cv::INTER_NEAREST);
    return big;
}

/**
 * @brief  在彩色图上标出所有命中位置。
 * @param  src   背景图。
 * @param  hits  击中击不中结果（255 为命中）。
 * @param  color 标记颜色。
 * @return 标注后的图像。
 */
cv::Mat markHits(const cv::Mat& src, const cv::Mat& hits, const cv::Scalar& color)
{
    cv::Mat view = src.clone();
    // 命中结果通常只有孤立的单个像素，先膨胀再画圈才看得见。
    cv::Mat dilated;
    cv::dilate(hits, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        const cv::Rect box = cv::boundingRect(contour);
        cv::circle(view, cv::Point(box.x + box.width / 2, box.y + box.height / 2), 26,
                   color, 2, cv::LINE_AA);
    }
    return view;
}

}  // namespace

/**
 * @brief  主流程：两类形状的匹配与标注。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/hit_or_miss.png }"
        "{s small        | 17   | 小方块的边长(像素) }"
        "{l large        | 35   | 大方块的边长(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("击中击不中演示: 用形态学核匹配特定尺寸的孤立形状\n参数请使用 --key=value 形式");
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

    // 步骤 1：定义击中/击不中核。素材是白底黑形状，先取反让形状成为前景。
    cv::Mat foreground;
    cv::threshold(src, foreground, 127, 255, cv::THRESH_BINARY_INV);
    const int smallSize = std::max(3, parser.get<int>("small"));
    const int largeSize = std::max(3, parser.get<int>("large"));
    const cv::Mat smallKernel = makeIsolationKernel(smallSize);
    const cv::Mat largeKernel = makeIsolationKernel(largeSize);

    // 步骤 2：应用变换。
    cv::Mat smallHits;
    cv::Mat largeHits;
    cv::morphologyEx(foreground, smallHits, cv::MORPH_HITMISS, smallKernel);
    cv::morphologyEx(foreground, largeHits, cv::MORPH_HITMISS, largeKernel);

    // 步骤 3：定位匹配。
    const int smallCount = cv::countNonZero(smallHits);
    const int largeCount = cv::countNonZero(largeHits);

    // 步骤 4：在彩色图上标注命中位置。
    const cv::Mat colorSrc = cv::imread(inputPath, cv::IMREAD_COLOR);
    const cv::Mat smallView =
        markHits(colorSrc, smallHits, cv::Scalar(0, 180, 0));
    const cv::Mat largeView =
        markHits(colorSrc, largeHits, cv::Scalar(0, 0, 255));

    std::cout << "输入: " << inputPath << "\n";
    std::cout << "  " << smallSize << "x" << smallSize << " 方块命中: " << smallCount << "\n";
    std::cout << "  " << largeSize << "x" << largeSize << " 方块命中: " << largeCount << "\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(foreground, "foreground (inverted)"),
        fitPanel(visualizeKernel(smallKernel, 12), "kernel " + std::to_string(smallSize) + "+ring"),
        fitPanel(smallView, "hits " + std::to_string(smallCount)),
        fitPanel(largeView, "hits " + std::to_string(largeCount))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/hit_or_miss.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("hit or miss", result);
        cv::waitKey(0);
    }
    return 0;
}
