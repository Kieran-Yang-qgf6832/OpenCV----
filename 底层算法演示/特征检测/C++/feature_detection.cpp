/**
 * @file    feature_detection.cpp
 * @brief   特征检测演示：多种检测器的关键点数量分布与小角度旋转下的重复率。
 * @details 流程：创建不同检测器 -> 检测关键点 -> 对比数量分布 -> 评估重复性。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          重复率的定义：把图像旋转 θ 后重新检测，把所有点映射回原坐标系，
 *          若距离原关键点 3 像素内存在对应点即视为重复。这个指标直接反映检测器的旋转不变性。
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
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
 * @brief  计算旋转后的重复率。
 * @param  detector  检测器。
 * @param  gray      灰度图。
 * @param  angleDeg  旋转角度（度）。
 * @param  tolerance 判定为重复的距离阈值（像素）。
 * @param  baseCount 输出：原图关键点数。
 * @return 重复率（0~1）。
 * @note   旋转后需要把新关键点用逆变换映射回原坐标系，再和原关键点比对。
 */
double repeatability(const cv::Ptr<cv::Feature2D>& detector, const cv::Mat& gray,
                     double angleDeg, double tolerance, int& baseCount)
{
    std::vector<cv::KeyPoint> basePoints;
    detector->detect(gray, basePoints);
    baseCount = static_cast<int>(basePoints.size());

    const cv::Point2f center(gray.cols / 2.0f, gray.rows / 2.0f);
    const cv::Mat rotation = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    cv::Mat rotated;
    cv::warpAffine(gray, rotated, rotation, gray.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);

    std::vector<cv::KeyPoint> rotatedPoints;
    detector->detect(rotated, rotatedPoints);
    if (basePoints.empty() || rotatedPoints.empty()) {
        return 0.0;
    }

    cv::Mat inverse;
    cv::invertAffineTransform(rotation, inverse);
    std::vector<cv::Point2f> rotatedCoordinates;
    rotatedCoordinates.reserve(rotatedPoints.size());
    for (const auto& kp : rotatedPoints) {
        rotatedCoordinates.push_back(kp.pt);
    }
    std::vector<cv::Point2f> mapped;
    cv::transform(rotatedCoordinates, mapped, inverse);

    int matched = 0;
    for (const auto& kp : basePoints) {
        for (const auto& pt : mapped) {
            if (cv::norm(kp.pt - pt) <= tolerance) {
                ++matched;
                break;
            }
        }
    }
    return static_cast<double>(matched) / static_cast<double>(basePoints.size());
}

}  // namespace

/**
 * @brief  主流程：四种检测器的数量对比与旋转重复率评估。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/building.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/feature_detection.png }"
        "{n maxFeatures  | 800  | 各检测器的特征点上限 }"
        "{a angle        | 5    | 重复性评估的旋转角度(度) }"
        "{t tolerance    | 3.0  | 重复判定距离阈值(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("特征检测演示: 多检测器数量分布与旋转重复率\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("building.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 1：创建不同检测器。
    const int maxFeatures = std::max(50, parser.get<int>("maxFeatures"));
    std::vector<std::pair<std::string, cv::Ptr<cv::Feature2D>>> detectors;
    detectors.emplace_back("ORB", cv::ORB::create(maxFeatures));
    detectors.emplace_back("SIFT", cv::SIFT::create(maxFeatures));
    detectors.emplace_back("AKAZE", cv::AKAZE::create());
    detectors.emplace_back("BRISK", cv::BRISK::create());

    std::vector<cv::Mat> panels = {fitPanel(gray, "gray input")};
    cv::RNG rng(20261119);
    std::cout << "输入: " << inputPath << ", 旋转 " << parser.get<double>("angle") << " 度\n";
    std::cout << std::left << std::setw(8) << "detector" << std::setw(12) << "keypoints"
              << "repeatability\n";

    for (const auto& entry : detectors) {
        // 步骤 2：检测关键点。
        std::vector<cv::KeyPoint> keypoints;
        entry.second->detect(gray, keypoints);

        // 步骤 3~4：对比数量并评估重复性。
        int baseCount = 0;
        const double rate = repeatability(entry.second, gray, parser.get<double>("angle"),
                                          parser.get<double>("tolerance"), baseCount);
        std::cout << std::left << std::setw(8) << entry.first << std::setw(12) << keypoints.size()
                  << std::fixed << std::setprecision(2) << rate * 100.0 << "%\n";

        cv::Mat view;
        cv::drawKeypoints(src, keypoints, view,
                          cv::Scalar(rng.uniform(60, 255), rng.uniform(60, 255), rng.uniform(60, 255)),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        char title[96];
        std::snprintf(title, sizeof(title), "%s: %zu pts, rep %.1f%%", entry.first.c_str(),
                      keypoints.size(), rate * 100.0);
        panels.push_back(fitPanel(view, title));
    }
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/feature_detection.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("feature detection", result);
        cv::waitKey(0);
    }
    return 0;
}
