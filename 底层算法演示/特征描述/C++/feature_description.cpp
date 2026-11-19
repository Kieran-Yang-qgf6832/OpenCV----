/**
 * @file    feature_description.cpp
 * @brief   特征描述演示：计算 ORB / SIFT 描述子，分析维度并检验旋转下的匹配能力。
 * @details 流程：检测关键点 -> 计算描述子 -> 分析维度 -> 可视化描述子并与旋转版本匹配。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          ORB 是 32 字节二进制描述子（汉明距离），SIFT 是 128 维浮点描述子（欧氏距离）；
 *          描述子的判别力要看「同一特征在变换后仍能匹配上」的比例，而不是维度高低。
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
 * @brief  把描述子矩阵渲染成可视化图像（每行一个描述子）。
 * @param  descriptors 描述子矩阵。
 * @param  maxRows     最多显示多少行。
 * @return 灰度可视化图，宽度已放大。
 */
cv::Mat visualizeDescriptors(const cv::Mat& descriptors, int maxRows)
{
    const int rows = std::min(maxRows, descriptors.rows);
    cv::Mat sliced = descriptors.rowRange(0, rows).clone();

    cv::Mat asFloat;
    sliced.convertTo(asFloat, CV_32F);
    cv::Mat normalized;
    cv::normalize(asFloat, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 描述子通常很窄（32 或 128 列），横向放大 4 倍才看得清每个比特/维度。
    cv::Mat enlarged;
    cv::resize(normalized, enlarged, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);
    return enlarged;
}

/**
 * @brief  统计旋转 15° 后的描述子匹配率。
 * @param  detector    检测器（同时提供描述子）。
 * @param  gray        灰度图。
 * @param  normType    距离度量。
 * @param  crossCheck  是否交叉验证（此处用比率筛选，故为 false）。
 * @param  goodRatio   输出：好匹配数。
 * @param  baseCount   输出：原图描述子数。
 * @param  matchesView 输出：匹配可视化。
 * @return 好匹配占原图描述子的比例。
 */
double rotationMatchRate(const cv::Ptr<cv::Feature2D>& detector, const cv::Mat& gray,
                         int normType, int& goodCount, int& baseCount, cv::Mat& matchesView)
{
    std::vector<cv::KeyPoint> kpBase;
    cv::Mat descBase;
    detector->detectAndCompute(gray, cv::noArray(), kpBase, descBase);

    const cv::Point2f center(gray.cols / 2.0f, gray.rows / 2.0f);
    cv::Mat rotated;
    cv::warpAffine(gray, rotated, cv::getRotationMatrix2D(center, 15.0, 1.0), gray.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    std::vector<cv::KeyPoint> kpRotated;
    cv::Mat descRotated;
    detector->detectAndCompute(rotated, cv::noArray(), kpRotated, descRotated);

    baseCount = descBase.rows;
    goodCount = 0;
    if (descBase.empty() || descRotated.empty()) {
        return 0.0;
    }

    cv::BFMatcher matcher(normType);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descBase, descRotated, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < 0.75 * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    goodCount = static_cast<int>(good.size());
    cv::drawMatches(gray, kpBase, rotated, kpRotated, good, matchesView,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return baseCount > 0 ? static_cast<double>(goodCount) / baseCount : 0.0;
}

}  // namespace

/**
 * @brief  主流程：两种描述子的维度分析与旋转匹配检验。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/church.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/feature_description.png }"
        "{n maxFeatures  | 500  | 特征点上限 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("特征描述演示: ORB / SIFT 描述子维度与旋转匹配率\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("church.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    const int maxFeatures = std::max(50, parser.get<int>("maxFeatures"));

    // 步骤 1~2：检测关键点并计算描述子。
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(maxFeatures);
    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(maxFeatures);

    std::vector<cv::KeyPoint> kpOrb;
    cv::Mat descOrb;
    orb->detectAndCompute(gray, cv::noArray(), kpOrb, descOrb);
    std::vector<cv::KeyPoint> kpSift;
    cv::Mat descSift;
    sift->detectAndCompute(gray, cv::noArray(), kpSift, descSift);

    // 步骤 3：分析维度。
    auto describe = [](const std::string& name, const cv::Mat& desc) {
        std::cout << "  " << std::left << std::setw(6) << name << " rows=" << std::setw(6)
                  << desc.rows << " cols=" << std::setw(5) << desc.cols
                  << " type=" << (desc.type() == CV_8U ? "CV_8U (二进制，汉明距离)"
                                                      : "CV_32F (浮点，欧氏距离)")
                  << "\n";
    };
    std::cout << "输入: " << inputPath << "\n";
    describe("ORB", descOrb);
    describe("SIFT", descSift);

    // 步骤 4：可视化描述子并做旋转匹配检验。
    int orbGood = 0;
    int orbBase = 0;
    cv::Mat orbMatches;
    const double orbRate = rotationMatchRate(orb, gray, cv::NORM_HAMMING, orbGood, orbBase,
                                             orbMatches);
    int siftGood = 0;
    int siftBase = 0;
    cv::Mat siftMatches;
    const double siftRate = rotationMatchRate(sift, gray, cv::NORM_L2, siftGood, siftBase,
                                              siftMatches);
    std::cout << "旋转 15° 后的匹配率:\n";
    std::cout << "  ORB : " << std::fixed << std::setprecision(1) << orbRate * 100.0
              << "% (" << orbGood << "/" << orbBase << ")\n";
    std::cout << "  SIFT: " << siftRate * 100.0 << "% (" << siftGood << "/" << siftBase << ")\n";

    cv::Mat orbView = src.clone();
    cv::drawKeypoints(src, kpOrb, orbView, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    const std::vector<cv::Mat> panels = {
        fitPanel(orbView, "ORB keypoints: " + std::to_string(kpOrb.size())),
        fitPanel(visualizeDescriptors(descOrb, 60),
                 "ORB descriptor " + std::to_string(descOrb.cols) + "x8bit"),
        fitPanel(visualizeDescriptors(descSift, 60),
                 "SIFT descriptor " + std::to_string(descSift.cols) + "xfloat"),
        fitPanel(orbMatches, "ORB rotated matches")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/feature_description.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("feature description", result);
        cv::waitKey(0);
    }
    return 0;
}
