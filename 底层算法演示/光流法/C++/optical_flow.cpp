/**
 * @file    optical_flow.cpp
 * @brief   光流法演示：Lucas-Kanade 稀疏光流与 Farneback 稠密光流。
 * @details 流程：读取连续两帧 -> 采样特征点 -> 计算光流矢量 -> 可视化运动。
 * @note    依赖：OpenCV 4.x（core / video / imgproc / imgcodecs / highgui）
 *          LK 假设「小位移 + 亮度恒定」，位移过大必须靠金字塔层（maxLevel）兜底；
 *          Farneback 给出逐像素位移场，用 HSV 编码（色相=方向，明度=幅值）最易读。
 */

#include <algorithm>
#include <cmath>
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
 * @brief  把稠密光流编码成 HSV 彩色图。
 * @param  flow 双通道浮点光流场 (dx, dy)。
 * @return BGR 可视化结果（色相表示方向，明度表示幅值）。
 */
cv::Mat flowToColor(const cv::Mat& flow)
{
    std::vector<cv::Mat> xy;
    cv::split(flow, xy);
    cv::Mat magnitude;
    cv::Mat angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);  // 角度单位为度

    cv::Mat hsv(flow.size(), CV_8UC3);
    std::vector<cv::Mat> hsvChannels(3);
    // 三个通道必须是同类型同尺寸，否则 merge 会断言失败。
    angle.convertTo(hsvChannels[0], CV_8U, 180.0 / 360.0);  // OpenCV 的 H 区间是 0~179
    hsvChannels[1] = cv::Mat(flow.size(), CV_8U, cv::Scalar(255));
    cv::normalize(magnitude, hsvChannels[2], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::merge(hsvChannels, hsv);

    cv::Mat color;
    cv::cvtColor(hsv, color, cv::COLOR_HSV2BGR);
    return color;
}

}  // namespace

/**
 * @brief  主流程：两帧之间的稀疏与稠密光流计算及可视化。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{a frameA       |      | 第一帧，缺省读取 assets/frame_01.png }"
        "{b frameB       |      | 第二帧，缺省读取 assets/frame_02.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/optical_flow.png }"
        "{p points       | 120  | LK 采样的最大特征点数 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("光流法演示: LK 稀疏光流 + Farneback 稠密光流\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string pathA = parser.get<std::string>("frameA");
    std::string pathB = parser.get<std::string>("frameB");
    if (pathA.empty()) {
        pathA = locateAsset("frame_01.png");
    }
    if (pathB.empty()) {
        pathB = locateAsset("frame_02.png");
    }

    // 步骤 1：读取连续两帧。
    const cv::Mat frameA = cv::imread(pathA, cv::IMREAD_COLOR);
    const cv::Mat frameB = cv::imread(pathB, cv::IMREAD_COLOR);
    if (frameA.empty() || frameB.empty()) {
        std::cerr << "无法读取连续帧: " << (frameA.empty() ? pathA : pathB) << std::endl;
        return -1;
    }
    cv::Mat grayA;
    cv::Mat grayB;
    cv::cvtColor(frameA, grayA, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameB, grayB, cv::COLOR_BGR2GRAY);

    // 步骤 2：采样特征点。角点比边缘更适合跟踪，因为后者存在孔径问题。
    std::vector<cv::Point2f> pointsA;
    cv::goodFeaturesToTrack(grayA, pointsA, std::max(4, parser.get<int>("points")), 0.01, 10.0);

    // 步骤 3：计算光流矢量。
    std::vector<cv::Point2f> pointsB;
    std::vector<uchar> status;
    std::vector<float> errors;
    if (!pointsA.empty()) {
        cv::calcOpticalFlowPyrLK(grayA, grayB, pointsA, pointsB, status, errors,
                                 cv::Size(21, 21), 3);
    }

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(grayA, grayB, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // 步骤 4：可视化运动。
    cv::Mat sparseView = frameA.clone();
    double sumShift = 0.0;
    int kept = 0;
    for (size_t i = 0; i < pointsA.size(); ++i) {
        if (!status.empty() && status[i] == 0) {
            continue;
        }
        cv::line(sparseView, pointsA[i], pointsB[i], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(sparseView, pointsB[i], 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        sumShift += cv::norm(pointsB[i] - pointsA[i]);
        ++kept;
    }

    const cv::Mat denseColor = flowToColor(flow);
    cv::Mat denseMagnitude;
    {
        std::vector<cv::Mat> xy;
        cv::split(flow, xy);
        cv::magnitude(xy[0], xy[1], denseMagnitude);
        cv::normalize(denseMagnitude, denseMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    // 稠密光流的中位位移，用来和稀疏跟踪的平均位移对照。
    cv::Scalar meanFlow = cv::mean(flow);
    std::cout << "LK 跟踪点: " << kept << "/" << pointsA.size()
              << ", 平均位移: " << (kept > 0 ? sumShift / kept : 0.0) << " 像素\n";
    std::cout << "Farneback 平均光流: dx=" << meanFlow[0] << ", dy=" << meanFlow[1] << "\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(frameA, "frame A"),
        fitPanel(sparseView, "LK sparse flow"),
        fitPanel(denseColor, "Farneback (HSV)"),
        fitPanel(denseMagnitude, "flow magnitude")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(pathA);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/optical_flow.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("optical flow", result);
        cv::waitKey(0);
    }
    return 0;
}
