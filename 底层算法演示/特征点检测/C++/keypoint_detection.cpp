/**
 * @file    keypoint_detection.cpp
 * @brief   特征点检测演示：ORB / SIFT / FAST 三种检测器的数量与耗时对比。
 * @details 流程：创建检测器 -> 检测关键点 -> 绘制关键点 -> 对比算法性能。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          FAST 只做检测、不做尺度与方向，数量最多但对尺度/旋转不具不变性；
 *          ORB = FAST + 方向 + 描述子，SIFT 有多尺度空间，数量与耗时都更高。
 */

#include <algorithm>
#include <chrono>
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
 * @brief  运行一个检测器并计时。
 * @param  detector   检测器对象。
 * @param  gray       灰度图。
 * @param  keypoints  输出：关键点。
 * @return 耗时（毫秒）。
 */
double runDetector(const cv::Ptr<cv::Feature2D>& detector, const cv::Mat& gray,
                   std::vector<cv::KeyPoint>& keypoints)
{
    const auto start = std::chrono::steady_clock::now();
    detector->detect(gray, keypoints);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

/**
 * @brief  主流程：三种检测器对比与可视化。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/books.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/keypoint_detection.png }"
        "{n maxFeatures  | 1000 | 各检测器的特征点上限 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("特征点检测演示: ORB / SIFT / FAST 对比\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("books.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 1：创建检测器。
    const int maxFeatures = std::max(50, parser.get<int>("maxFeatures"));
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(maxFeatures);
    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(maxFeatures);
    // FAST 没有数量上限，阈值越低点越多。
    const cv::Ptr<cv::FastFeatureDetector> fast =
        cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);

    // 步骤 2~3：检测并绘制。
    struct Result {
        std::string name;
        std::vector<cv::KeyPoint> keypoints;
        double milliseconds;
        cv::Mat view;
    };
    std::vector<Result> results;

    auto add = [&](const std::string& name, const cv::Ptr<cv::Feature2D>& detector,
                   const cv::Scalar& color) {
        Result r;
        r.name = name;
        r.milliseconds = runDetector(detector, gray, r.keypoints);
        r.view = src.clone();
        cv::drawKeypoints(src, r.keypoints, r.view, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        results.push_back(r);
    };
    add("ORB", orb, cv::Scalar(0, 255, 0));
    add("SIFT", sift, cv::Scalar(0, 200, 255));
    add("FAST", fast, cv::Scalar(255, 0, 255));

    // 步骤 4：对比算法性能。
    std::cout << "输入: " << inputPath << " (" << src.cols << "x" << src.rows << ")\n";
    std::cout << std::left << std::setw(10) << "detector" << std::setw(14) << "keypoints"
              << "time(ms)\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(8) << r.name << std::setw(12) << r.keypoints.size()
                  << std::fixed << std::setprecision(2) << r.milliseconds << "\n";
    }

    std::vector<cv::Mat> panels;
    panels.push_back(fitPanel(gray, "gray input"));
    for (const auto& r : results) {
        panels.push_back(fitPanel(r.view, r.name + ": " + std::to_string(r.keypoints.size())));
    }
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/keypoint_detection.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("keypoint detection", result);
        cv::waitKey(0);
    }
    return 0;
}
