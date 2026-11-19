/**
 * @file    akaze_tracking.cpp
 * @brief   AKAZE 跟踪演示：以首帧为参考，逐帧匹配并估计单应，绘制跟踪框。
 * @details 流程：初始化参考帧 -> 逐帧匹配 -> 估计变换 -> 绘制轨迹并写出视频。
 * @note    依赖：OpenCV 4.x（core / features2d / calib3d / videoio / imgproc / highgui）
 *          匹配 + 单应比逐帧光流更抗尺度变化：本素材镜头在平移同时轻微缩小，
 *          用相似变换（单应）能把缩放一并吸收，框不会随时间漂移出画面。
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
 * @brief  在待匹配帧上估计参考框的新位置并绘制。
 * @param  detector    AKAZE 检测器。
 * @param  refPoints   参考帧的关键点坐标。
 * @param  refDesc     参考帧描述子。
 * @param  refCorners  参考框的四个角。
 * @param  frame       当前帧（灰度）。
 * @param  colorFrame  当前帧（彩色，用于绘制）。
 * @param  ratio       Lowe 比率阈值。
 * @return 当前帧的跟踪框；失败时返回空 vector。
 */
std::vector<cv::Point2f> trackQuad(const cv::Ptr<cv::AKAZE>& detector,
                                   const std::vector<cv::Point2f>& refPoints,
                                   const cv::Mat& refDesc,
                                   const std::vector<cv::Point2f>& refCorners,
                                   const cv::Mat& frame, cv::Mat& colorFrame, double ratio)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);
    if (descriptors.empty()) {
        return {};
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(refDesc, descriptors, knn, 2);

    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            src.push_back(refPoints[static_cast<size_t>(pair[0].queryIdx)]);
            dst.push_back(keypoints[static_cast<size_t>(pair[0].trainIdx)].pt);
        }
    }
    if (src.size() < 8) {
        return {};
    }

    const cv::Mat homography = cv::findHomography(src, dst, cv::RANSAC, 3.0);
    if (homography.empty()) {
        return {};
    }
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(refCorners, projected, homography);
    return projected;
}

}  // namespace

/**
 * @brief  主流程：逐帧跟踪并输出视频与采样帧拼接图。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 视频或首帧图像，缺省读取 assets/tracking.mp4 }"
        "{o output       |      | 输出目录，缺省写入 <目录>/output }"
        "{r ratio        | 0.75 | Lowe 比率阈值 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("AKAZE 跟踪演示: 首帧参考 + 逐帧匹配 + 单应估计\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("tracking.mp4");
    }
    std::string outDir = parser.get<std::string>("output");
    if (outDir.empty()) {
        outDir = outputDirFor(inputPath);
    }
    cv::utils::fs::createDirectories(outDir);
    const double ratio = parser.get<double>("ratio");

    // 步骤 1：初始化参考帧（视频首帧）。
    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "无法打开视频: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat reference;
    if (!capture.read(reference)) {
        std::cerr << "无法读取首帧" << std::endl;
        return -1;
    }
    const double fps = capture.get(cv::CAP_PROP_FPS) > 0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    const cv::Size frameSize = reference.size();

    const cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    std::vector<cv::KeyPoint> refKeypoints;
    cv::Mat refDesc;
    cv::Mat refGray;
    cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);
    detector->detectAndCompute(refGray, cv::noArray(), refKeypoints, refDesc);
    if (refDesc.empty()) {
        std::cerr << "首帧特征提取失败" << std::endl;
        return -1;
    }
    std::vector<cv::Point2f> refPoints;
    refPoints.reserve(refKeypoints.size());
    for (const auto& kp : refKeypoints) {
        refPoints.push_back(kp.pt);
    }
    // 参考框取整幅画面，跟踪时它的形变就是相机的整体运动。
    const std::vector<cv::Point2f> refCorners = {
        {0.0f, 0.0f},
        {static_cast<float>(frameSize.width), 0.0f},
        {static_cast<float>(frameSize.width), static_cast<float>(frameSize.height)},
        {0.0f, static_cast<float>(frameSize.height)}};

    cv::VideoWriter writer(outDir + "/akaze_tracking.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频" << std::endl;
        return -1;
    }

    // 步骤 2~4：逐帧匹配、估计变换、绘制轨迹。
    std::vector<cv::Mat> samples;
    cv::Mat frame;
    int index = 0;
    int tracked = 0;
    while (capture.read(frame)) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat canvas = frame.clone();
        const std::vector<cv::Point2f> quad =
            trackQuad(detector, refPoints, refDesc, refCorners, gray, canvas, ratio);
        if (!quad.empty()) {
            ++tracked;
            std::vector<cv::Point> polygon;
            for (const auto& pt : quad) {
                polygon.emplace_back(cvRound(pt.x), cvRound(pt.y));
            }
            cv::polylines(canvas, std::vector<std::vector<cv::Point>>{polygon}, true,
                          cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }
        cv::putText(canvas, "frame " + std::to_string(index), cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        writer.write(canvas);

        if (index % 25 == 0) {
            samples.push_back(canvas.clone());
        }
        if (parser.get<int>("display") != 0) {
            cv::imshow("akaze tracking", canvas);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        ++index;
    }
    writer.release();

    std::cout << "输入: " << inputPath << ", 处理 " << index << " 帧, 成功跟踪 " << tracked
              << " 帧 (" << (index > 0 ? 100.0 * tracked / index : 0.0) << "%)\n";
    std::cout << "  参考帧关键点: " << refKeypoints.size() << "\n";
    std::cout << "  跟踪视频: " << outDir << "/akaze_tracking.mp4\n";

    std::vector<cv::Mat> panels;
    for (size_t i = 0; i < samples.size() && i < 4; ++i) {
        panels.push_back(fitPanel(samples[i], "frame " + std::to_string(i * 25)));
    }
    if (panels.empty()) {
        panels.push_back(fitPanel(reference, "reference (no frame read)"));
    }
    const cv::Mat result = montage(panels, 2);
    const std::string outputPath = outDir + "/akaze_tracking.png";
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "  采样帧拼接: " << outputPath << std::endl;
    return 0;
}
