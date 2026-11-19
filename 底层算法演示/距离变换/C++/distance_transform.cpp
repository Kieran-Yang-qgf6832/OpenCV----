/**
 * @file    distance_transform.cpp
 * @brief   距离变换演示：计算每个前景像素到最近背景像素的距离，并对比三种距离度量。
 * @details 流程：二值化 -> 计算距离变换 -> 归一化显示 -> 由距离峰值提取前景种子（配合分水岭）。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          DIST_L2 给出欧氏距离，是对「物体粗细」最直观的度量；
 *          DIST_L1 是城市街区距离，呈菱形扩散；DIST_C 是棋盘距离，呈方形扩散。
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
 * @brief  把浮点距离图归一化后上色，便于观察距离分布。
 * @param  distance 距离图（CV_32F）。
 * @return BGR 伪彩色图。
 */
cv::Mat distanceView(const cv::Mat& distance)
{
    cv::Mat normalized;
    cv::normalize(distance, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    return colored;
}

}  // namespace

/**
 * @brief  主流程：三种距离度量对比，并由此提取分水岭种子。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/distance_transform.png }"
        "{s seedRatio    | 0.5  | 前景种子的阈值，占最大距离的比例 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("距离变换演示: DIST_L1 / DIST_L2 / DIST_C 对比与种子提取\n参数请使用 --key=value 形式");
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

    // 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景（距离变换只对非零像素计算）。
    cv::Mat foreground;
    cv::threshold(src, foreground, 127, 255, cv::THRESH_BINARY_INV);

    // 步骤 2：计算距离变换。第三个参数是掩膜尺寸，3 或 5 表示使用 3x3 / 5x5 邻域。
    cv::Mat distanceL1;
    cv::Mat distanceL2;
    cv::Mat distanceC;
    cv::distanceTransform(foreground, distanceL1, cv::DIST_L1, 3);
    cv::distanceTransform(foreground, distanceL2, cv::DIST_L2, 5);
    cv::distanceTransform(foreground, distanceC, cv::DIST_C, 3);

    // 步骤 3：归一化显示。
    const cv::Mat viewL1 = distanceView(distanceL1);
    const cv::Mat viewL2 = distanceView(distanceL2);
    const cv::Mat viewC = distanceView(distanceC);

    // 步骤 4：结合分水岭。距离峰值即各物体的中心，阈值化后可作为前景种子。
    double maxL2 = 0.0;
    cv::minMaxLoc(distanceL2, nullptr, &maxL2);
    cv::Mat seeds;
    cv::threshold(distanceL2, seeds, parser.get<double>("seedRatio") * maxL2, 255,
                  cv::THRESH_BINARY);
    seeds.convertTo(seeds, CV_8U);
    cv::Mat seedLabels;
    const int seedCount = cv::connectedComponents(seeds, seedLabels) - 1;

    double maxL1 = 0.0;
    double maxC = 0.0;
    cv::minMaxLoc(distanceL1, nullptr, &maxL1);
    cv::minMaxLoc(distanceC, nullptr, &maxC);
    std::cout << "输入: " << inputPath << "\n";
    std::cout << "  DIST_L1 最大距离: " << std::fixed << std::setprecision(2) << maxL1 << "\n";
    std::cout << "  DIST_L2 最大距离: " << maxL2 << "\n";
    std::cout << "  DIST_C  最大距离: " << maxC << "\n";
    std::cout << "  阈值 " << parser.get<double>("seedRatio") << "*max 提取的种子数: "
              << seedCount << "\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(foreground, "foreground (inverted)"),
        fitPanel(viewL1, "DIST_L1 (manhattan)"),
        fitPanel(viewL2, "DIST_L2 (euclidean)"),
        fitPanel(viewC, "DIST_C (chessboard)")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/distance_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("distance transform", result);
        cv::waitKey(0);
    }
    return 0;
}
