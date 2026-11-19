/**
 * @file    flann_matching.cpp
 * @brief   FLANN 匹配器演示：用 KD 树索引做快速近似最近邻匹配，并与暴力匹配对比耗时。
 * @details 流程：选择索引算法 -> 设置搜索参数 -> kNN 匹配 -> 比率测试筛选。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          FLANN 是「近似」最近邻：用少量精度换取速度，描述子维度越高、数据量越大收益越明显；
 *          SIFT 的浮点描述子用 KDTreeIndexParams，ORB/AKAZE 的二进制描述子需改用 LSH 索引。
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
 * @brief  用比率测试从 knn 结果里筛出好匹配。
 * @param  knn   每个查询点的两个最近邻。
 * @param  ratio 比率阈值。
 * @return 好匹配列表。
 */
std::vector<cv::DMatch> ratioTest(const std::vector<std::vector<cv::DMatch>>& knn, double ratio)
{
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

}  // namespace

/**
 * @brief  主流程：FLANN 与 BFMatcher 的匹配质量与耗时对比。
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
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/flann_matching.png }"
        "{r ratio        | 0.75 | Lowe 比率阈值 }"
        "{t trees        | 5    | KD 树数量 }"
        "{c checks       | 50   | 搜索时检查的叶节点数，越大越精确 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("FLANN 匹配器演示: KDTree 索引 + 近似 kNN\n参数请使用 --key=value 形式");
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

    // 步骤 1：用 SIFT 得到浮点描述子（FLANN 的 KD 树适用于浮点特征）。
    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(1500);
    std::vector<cv::KeyPoint> kpA;
    std::vector<cv::KeyPoint> kpB;
    cv::Mat descA;
    cv::Mat descB;
    sift->detectAndCompute(imageA, cv::noArray(), kpA, descA);
    sift->detectAndCompute(imageB, cv::noArray(), kpB, descB);
    if (descA.empty() || descB.empty()) {
        std::cerr << "特征提取失败" << std::endl;
        return -1;
    }
    const cv::Mat descAFloat = descA;
    const cv::Mat descBFloat = descB;

    // 步骤 2：设置 FLANN 索引与搜索参数。
    const int trees = std::max(1, parser.get<int>("trees"));
    const int checks = std::max(1, parser.get<int>("checks"));
    cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::KDTreeIndexParams>(trees),
                                cv::makePtr<cv::flann::SearchParams>(checks));

    // 步骤 3：kNN 匹配，同时用暴力匹配作对照。
    const double ratio = parser.get<double>("ratio");
    std::vector<std::vector<cv::DMatch>> knnFlann;
    std::vector<std::vector<cv::DMatch>> knnBrute;
    const int64 t0 = cv::getTickCount();
    flann.knnMatch(descAFloat, descBFloat, knnFlann, 2);
    const int64 t1 = cv::getTickCount();
    cv::BFMatcher brute(cv::NORM_L2);
    brute.knnMatch(descAFloat, descBFloat, knnBrute, 2);
    const int64 t2 = cv::getTickCount();
    const double flannMs = (t1 - t0) * 1000.0 / cv::getTickFrequency();
    const double bruteMs = (t2 - t1) * 1000.0 / cv::getTickFrequency();

    // 步骤 4：比率测试筛选并可视化。
    const std::vector<cv::DMatch> goodFlann = ratioTest(knnFlann, ratio);
    const std::vector<cv::DMatch> goodBrute = ratioTest(knnBrute, ratio);

    cv::Mat flannView;
    cv::Mat bruteView;
    cv::drawMatches(imageA, kpA, imageB, kpB, goodFlann, flannView, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(imageA, kpA, imageB, kpB, goodBrute, bruteView, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::cout << "描述子: A " << descA.rows << "x" << descA.cols << ", B " << descB.rows << "x"
              << descB.cols << "\n";
    std::cout << std::left << std::setw(10) << "matcher" << std::setw(12) << "good"
              << "time(ms)\n";
    std::cout << std::left << std::setw(10) << "FLANN" << std::setw(12) << goodFlann.size()
              << std::fixed << std::setprecision(2) << flannMs << "\n";
    std::cout << std::left << std::setw(10) << "BFMatcher" << std::setw(12) << goodBrute.size()
              << bruteMs << "\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(flannView, "FLANN: " + std::to_string(goodFlann.size())),
        fitPanel(bruteView, "BFMatcher: " + std::to_string(goodBrute.size()))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(pathA);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/flann_matching.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("flann matching", result);
        cv::waitKey(0);
    }
    return 0;
}
