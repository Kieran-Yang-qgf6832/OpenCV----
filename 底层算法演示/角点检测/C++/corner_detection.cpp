/**
 * @file    corner_detection.cpp
 * @brief   角点检测演示：Harris 响应与 Shi-Tomasi（goodFeaturesToTrack）对比。
 * @details 流程：灰度转换 -> 计算角点响应 -> 阈值筛选 -> 非极大值抑制 -> 标注角点。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          Harris 用响应图 R = det(M) - k·trace(M)²，需要自己阈值 + 非极大值抑制；
 *          Shi-Tomasi 直接取 M 的较小特征值，goodFeaturesToTrack 内部已做 NMS 与最小间距。
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
 * @brief  Harris 角点检测：响应阈值 + 非极大值抑制。
 * @param  gray      灰度图。
 * @param  threshold 归一化后响应的相对阈值（占最大值的比例）。
 * @return 角点列表。
 */
std::vector<cv::Point> harrisCorners(const cv::Mat& gray, double threshold)
{
    cv::Mat response;
    cv::cornerHarris(gray, response, 2, 3, 0.04);
    cv::Mat normalized;
    cv::normalize(response, normalized, 0, 1, cv::NORM_MINMAX, CV_32F);

    // 非极大值抑制：只有本像素等于邻域最大值时才保留，避免角点成片。
    // 窗口取 9x9：砖墙纹理极密，3x3 窗口压不住，会留下上万个角点。
    cv::Mat dilated;
    cv::dilate(normalized, dilated,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    cv::Mat mask = (normalized > threshold) & (normalized >= dilated);

    std::vector<cv::Point> corners;
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uchar>(y, x) != 0) {
                corners.emplace_back(x, y);
            }
        }
    }
    return corners;
}

}  // namespace

/**
 * @brief  主流程：两种角点检测算法的结果对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/brick.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/corner_detection.png }"
        "{t harrisThresh | 0.5  | Harris 相对阈值（占归一化响应的比例） }"
        "{m maxCorners   | 200  | Shi-Tomasi 最大角点数 }"
        "{q minDistance  | 10   | Shi-Tomasi 角点最小间距 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("角点检测演示: Harris 与 Shi-Tomasi 对比\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("brick.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：灰度转换。
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 2~4：Harris 响应、阈值、非极大值抑制。
    const std::vector<cv::Point> harris = harrisCorners(gray, parser.get<double>("harrisThresh"));
    cv::Mat harrisView = src.clone();
    for (const auto& pt : harris) {
        cv::circle(harrisView, pt, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    cv::Mat responseView;
    {
        cv::Mat response;
        cv::cornerHarris(gray, response, 2, 3, 0.04);
        cv::normalize(response, response, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::Mat response8;
        response.convertTo(response8, CV_8U);
        cv::applyColorMap(response8, responseView, cv::COLORMAP_JET);
    }

    // Shi-Tomasi：goodFeaturesToTrack 内部已完成质量筛选与最小间距抑制。
    std::vector<cv::Point2f> shiTomasi;
    cv::goodFeaturesToTrack(gray, shiTomasi, std::max(1, parser.get<int>("maxCorners")),
                            parser.get<double>("harrisThresh"), parser.get<double>("minDistance"));
    cv::Mat shiView = src.clone();
    for (const auto& pt : shiTomasi) {
        cv::circle(shiView, pt, 4, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    std::cout << "输入: " << inputPath << " (" << gray.cols << "x" << gray.rows << ")\n";
    std::cout << "Harris 角点数: " << harris.size()
              << " (阈值 " << parser.get<double>("harrisThresh") << ")\n";
    std::cout << "Shi-Tomasi 角点数: " << shiTomasi.size()
              << " (上限 " << parser.get<int>("maxCorners") << ")\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(responseView, "harris response"),
        fitPanel(harrisView, "harris NMS: " + std::to_string(harris.size())),
        fitPanel(shiView, "shi-tomasi: " + std::to_string(shiTomasi.size()))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/corner_detection.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("corner detection", result);
        cv::waitKey(0);
    }
    return 0;
}
