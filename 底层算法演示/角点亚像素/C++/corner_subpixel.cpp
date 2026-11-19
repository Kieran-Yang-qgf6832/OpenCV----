/**
 * @file    corner_subpixel.cpp
 * @brief   角点亚像素演示：把整数像素角点精化到亚像素精度，并检验收敛性。
 * @details 流程：粗检测角点 -> 设定搜索窗口 -> cornerSubPix 精化 -> 与抖动初值的结果对比精度。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          cornerSubPix 基于梯度正交假设迭代求解：所有边缘方向的梯度都应垂直于
 *          「角点→邻域点」向量。它要求输入图像有抗锯齿或真实成像的灰度过渡，
 *          纯二值图（只有 0/255）会退化成一个固定的半像素偏置。
 */

#include <algorithm>
#include <cmath>
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
 * @brief  对给定初值做亚像素精化。
 * @param  gray     灰度图。
 * @param  corners  初值（会被修改为精化结果）。
 * @param  winSize  搜索窗口半径。
 * @param  maxIter  最大迭代次数。
 * @param  epsilon  收敛阈值。
 */
void refineCorners(const cv::Mat& gray, std::vector<cv::Point2f>& corners,
                   int winSize, int maxIter, double epsilon)
{
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                    maxIter, epsilon);
    cv::cornerSubPix(gray, corners, cv::Size(winSize, winSize), cv::Size(-1, -1), criteria);
}

/**
 * @brief  计算梯度正交残差：窗口内 sum |∇I(q)·(q-p)|² / N。
 * @param  gx    x 方向梯度（CV_32F）。
 * @param  gy    y 方向梯度（CV_32F）。
 * @param  p     角点位置。
 * @param  half  窗口半径。
 * @return 平均残差，越小说明角点位置越符合「梯度垂直于角点→邻域点向量」的假设。
 * @note   这正是 cornerSubPix 最小化的目标函数，可直接用来量化精化效果。
 */
double orthogonalityResidual(const cv::Mat& gx, const cv::Mat& gy,
                             const cv::Point2f& p, int half)
{
    const int x0 = cvRound(p.x);
    const int y0 = cvRound(p.y);
    double total = 0.0;
    int count = 0;
    for (int y = y0 - half; y <= y0 + half; ++y) {
        for (int x = x0 - half; x <= x0 + half; ++x) {
            if (x < 0 || y < 0 || x >= gx.cols || y >= gx.rows) {
                continue;
            }
            const double dx = x - p.x;
            const double dy = y - p.y;
            const double value = gx.at<float>(y, x) * dx + gy.at<float>(y, x) * dy;
            total += value * value;
            ++count;
        }
    }
    return count > 0 ? total / count : 0.0;
}

/**
 * @brief  计算一组角点的平均正交残差。
 * @param  gray      灰度图。
 * @param  corners   角点位置。
 * @param  half      窗口半径。
 * @return 平均残差。
 */
double meanResidual(const cv::Mat& gray, const std::vector<cv::Point2f>& corners, int half)
{
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    double total = 0.0;
    for (const auto& pt : corners) {
        total += orthogonalityResidual(gx, gy, pt, half);
    }
    return corners.empty() ? 0.0 : total / static_cast<double>(corners.size());
}

/** @brief 计算两组点之间的平均/最大距离。 */
void printError(const std::string& name, const std::vector<cv::Point2f>& a,
                const std::vector<cv::Point2f>& b)
{
    double sum = 0.0;
    double maxValue = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        const double d = cv::norm(a[i] - b[i]);
        sum += d;
        maxValue = std::max(maxValue, d);
    }
    const double mean = a.empty() ? 0.0 : sum / static_cast<double>(a.size());
    std::cout << "  " << std::left << std::setw(28) << name
              << " mean=" << std::fixed << std::setprecision(4) << std::setw(8) << mean
              << " max=" << std::setw(8) << maxValue << "\n";
}

}  // namespace

/**
 * @brief  主流程：粗检测、亚像素精化、抖动初值收敛性检验。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/checkerboard.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/corner_subpixel.png }"
        "{n maxCorners   | 200  | 粗检测角点上限 }"
        "{q quality      | 0.40 | 角点质量阈值，太低会把棋盘外边界的弱角点也收进来 }"
        "{m minDistance  | 60   | 角点最小间距，应大于棋盘格的半格宽 }"
        "{w winSize      | 5    | cornerSubPix 搜索窗口半径 }"
        "{j jitter       | 1.5  | 收敛性检验时给初值加的最大抖动(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("角点亚像素演示: cornerSubPix 精化与收敛性\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("checkerboard.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 1：粗检测角点（整数坐标）。
    std::vector<cv::Point2f> coarse;
    cv::goodFeaturesToTrack(gray, coarse, std::max(1, parser.get<int>("maxCorners")),
                            parser.get<double>("quality"), parser.get<double>("minDistance"));
    if (coarse.empty()) {
        std::cerr << "未检测到角点" << std::endl;
        return -1;
    }

    // 步骤 2~3：设定搜索窗口并做亚像素精化。
    const int winSize = std::max(2, parser.get<int>("winSize"));
    std::vector<cv::Point2f> refined = coarse;
    refineCorners(gray, refined, winSize, 40, 0.001);

    // 步骤 4：对比精度。给初值加上随机抖动再精化，看结果是否收敛到同一位置。
    const double jitter = parser.get<double>("jitter");
    cv::RNG rng(20261119);
    std::vector<cv::Point2f> jittered = coarse;
    for (auto& pt : jittered) {
        pt.x += static_cast<float>(rng.uniform(-jitter, jitter));
        pt.y += static_cast<float>(rng.uniform(-jitter, jitter));
    }
    std::vector<cv::Point2f> jitterRefined = jittered;
    refineCorners(gray, jitterRefined, winSize, 40, 0.001);

    std::cout << "输入: " << inputPath << ", 角点数: " << coarse.size()
              << ", 搜索窗口: " << winSize << "x" << winSize << "\n";
    printError("整数 -> 亚像素", coarse, refined);
    printError("抖动初值 -> 亚像素", refined, jitterRefined);

    // 梯度正交残差：cornerSubPix 的目标函数值，越小说明定位越准。
    const double residualCoarse = meanResidual(gray, coarse, winSize);
    const double residualRefined = meanResidual(gray, refined, winSize);
    std::cout << "  梯度正交残差: 整数 " << std::fixed << std::setprecision(0) << residualCoarse
              << " -> 亚像素 " << residualRefined << " ("
              << std::setprecision(1)
              << 100.0 * (1.0 - residualRefined / std::max(1e-9, residualCoarse)) << "% 改善)\n";

    // 可视化：整数角点（绿）与亚像素角点（红）。
    cv::Mat coarseView = src.clone();
    for (const auto& pt : coarse) {
        cv::circle(coarseView, pt, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    cv::Mat refinedView = src.clone();
    for (size_t i = 0; i < refined.size(); ++i) {
        cv::circle(refinedView, coarse[i], 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::drawMarker(refinedView, refined[i], cv::Scalar(0, 0, 255),
                       cv::MARKER_CROSS, 9, 1, cv::LINE_AA);
    }

    // 放大一个角点，肉眼才能看出半像素级的差别。
    const cv::Point2f probe = refined[refined.size() / 2];
    const int half = 12;
    const cv::Rect zoomRect(
        std::max(0, static_cast<int>(probe.x) - half),
        std::max(0, static_cast<int>(probe.y) - half), 2 * half, 2 * half);
    cv::Mat zoom = refinedView(zoomRect).clone();
    cv::resize(zoom, zoom, cv::Size(zoom.cols * 12, zoom.rows * 12), 0, 0, cv::INTER_NEAREST);

    const std::vector<cv::Mat> panels = {
        fitPanel(coarseView, "integer corners (green)"),
        fitPanel(refinedView, "sub-pixel (red cross)"),
        fitPanel(zoom, "zoom x12")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/corner_subpixel.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("corner subpixel", result);
        cv::waitKey(0);
    }
    return 0;
}
