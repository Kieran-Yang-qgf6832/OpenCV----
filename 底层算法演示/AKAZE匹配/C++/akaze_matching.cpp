/**
 * @file    akaze_matching.cpp
 * @brief   AKAZE 特征匹配演示：检测关键点与描述子，用 BFMatcher + 比率测试筛选。
 * @details 流程：检测关键点和描述子 -> BFMatcher 匹配 -> 比率测试筛选 -> 可视化。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          AKAZE 基于非线性尺度空间，描述子为二进制（MLDB），因此匹配用汉明距离；
 *          比率测试比较「最近邻」与「次近邻」，能挡掉大量只有唯一候选的误匹配。
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

}  // namespace

/**
 * @brief  主流程：AKAZE 匹配与筛选可视化。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{a imageA       |      | 图像 A，缺省读取 assets/scene.png }"
        "{b imageB       |      | 图像 B，缺省读取 assets/scene_rotated.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/akaze_matching.png }"
        "{r ratio        | 0.75 | Lowe 比率阈值 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("AKAZE 匹配演示: 检测 + BFMatcher + 比率测试\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string pathA = parser.get<std::string>("imageA");
    std::string pathB = parser.get<std::string>("imageB");
    if (pathA.empty()) {
        pathA = locateAsset("scene.png");
    }
    if (pathB.empty()) {
        pathB = locateAsset("scene_rotated.png");
    }

    const cv::Mat imageA = cv::imread(pathA, cv::IMREAD_COLOR);
    const cv::Mat imageB = cv::imread(pathB, cv::IMREAD_COLOR);
    if (imageA.empty() || imageB.empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return -1;
    }

    // 步骤 1：检测关键点和描述子。
    const cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> kpA;
    std::vector<cv::KeyPoint> kpB;
    cv::Mat descA;
    cv::Mat descB;
    akaze->detectAndCompute(imageA, cv::noArray(), kpA, descA);
    akaze->detectAndCompute(imageB, cv::noArray(), kpB, descB);
    if (descA.empty() || descB.empty()) {
        std::cerr << "特征提取失败" << std::endl;
        return -1;
    }

    // 步骤 2~3：BFMatcher + 比率测试。
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descA, descB, knn, 2);
    const double ratio = parser.get<double>("ratio");
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }

    // 步骤 4：可视化。分「全部 knn 首选」与「比率筛选后」两格对比。
    std::vector<cv::DMatch> allFirst;
    allFirst.reserve(knn.size());
    for (const auto& pair : knn) {
        if (!pair.empty()) {
            allFirst.push_back(pair[0]);
        }
    }
    cv::Mat allView;
    cv::Mat goodView;
    cv::drawMatches(imageA, kpA, imageB, kpB, allFirst, allView, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(imageA, kpA, imageB, kpB, good, goodView, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::cout << "A 关键点: " << kpA.size() << ", B 关键点: " << kpB.size()
              << ", 描述子维度: " << descA.cols << " 字节\n";
    std::cout << "knn 首选匹配: " << allFirst.size() << ", 比率 " << ratio << " 筛选后: "
              << good.size() << " (" << std::fixed << std::setprecision(1)
              << 100.0 * good.size() / std::max<size_t>(1, allFirst.size()) << "%)\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(imageA, "image A"),
        fitPanel(imageB, "image B (rotated)"),
        fitPanel(allView, "knn first: " + std::to_string(allFirst.size())),
        fitPanel(goodView, "ratio " + std::to_string(ratio).substr(0, 4) + ": " +
                               std::to_string(good.size()))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(pathA);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/akaze_matching.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("akaze matching", result);
        cv::waitKey(0);
    }
    return 0;
}
