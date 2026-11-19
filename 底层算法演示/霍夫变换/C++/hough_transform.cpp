/**
 * @file    hough_transform.cpp
 * @brief   霍夫变换演示：标准霍夫直线变换与概率霍夫直线变换。
 * @details 流程：边缘检测 -> 霍夫空间累加 -> 峰值检测 -> 绘制直线/线段。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          标准 HoughLines 输出 (rho, theta) 参数对，得到的是贯穿全图的直线；
 *          HoughLinesP 直接给出线段端点，并且并行累计、速度更快，工程上更常用。
 *          霍夫圆需要画面中存在圆形目标，本素材（铁路）只有直线，故只演示直线。
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

}  // namespace

/**
 * @brief  主流程：边缘检测 + 两种霍夫直线变换。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/railway.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/hough_transform.png }"
        "{t linesThresh  | 400  | 标准霍夫的累加器阈值 }"
        "{p pThresh      | 180  | 概率霍夫的累加器阈值 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("霍夫变换演示: HoughLines + HoughLinesP\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("railway.png");
    }

    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：边缘检测。霍夫变换的输入必须是边缘图，阈值的选取直接影响结果数量。
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3);

    // 步骤 2~4：累加、取峰值并绘制。标准霍夫得到的是无限长直线的参数对。
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1.0, CV_PI / 180.0, parser.get<double>("linesThresh"));
    cv::Mat lineView = src.clone();
    for (const auto& line : lines) {
        const float rho = line[0];
        const float theta = line[1];
        const double cosT = std::cos(theta);
        const double sinT = std::sin(theta);
        const double x0 = cosT * rho;
        const double y0 = sinT * rho;
        // 沿直线方向取足够长的两点，保证画到画面之外。
        const cv::Point p1(cvRound(x0 + 2000 * (-sinT)), cvRound(y0 + 2000 * cosT));
        const cv::Point p2(cvRound(x0 - 2000 * (-sinT)), cvRound(y0 - 2000 * cosT));
        cv::line(lineView, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    std::vector<cv::Vec4i> segments;
    cv::HoughLinesP(edges, segments, 1.0, CV_PI / 180.0, parser.get<double>("pThresh"), 150.0, 8.0);
    cv::Mat segmentView = src.clone();
    for (const auto& seg : segments) {
        cv::line(segmentView, cv::Point(seg[0], seg[1]), cv::Point(seg[2], seg[3]),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    std::cout << "标准霍夫直线数: " << lines.size() << "\n";
    std::cout << "概率霍夫线段数: " << segments.size() << "\n";
    for (size_t i = 0; i < lines.size() && i < 5; ++i) {
        std::cout << "  rho=" << std::fixed << std::setprecision(1) << lines[i][0]
                  << ", theta=" << lines[i][1] * 180.0 / CV_PI << " 度\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(edges, "canny edges"),
        fitPanel(lineView, "HoughLines: " + std::to_string(lines.size())),
        fitPanel(segmentView, "HoughLinesP: " + std::to_string(segments.size()))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/hough_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("hough transform", result);
        cv::waitKey(0);
    }
    return 0;
}
