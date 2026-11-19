/**
 * @file    edge_detection.cpp
 * @brief   边缘检测演示：Sobel、Laplacian 与 Canny 三种算子的对比。
 * @details 流程：灰度转换 -> 高斯降噪 -> 计算梯度幅值 / 二阶导 -> 双阈值处理 -> 输出边缘图。
 * @note    依赖：OpenCV 4.x（core / imgproc / videoio / imgcodecs / highgui）
 *          Sobel 与 Laplacian 都只是「响应」，还需要自己阈值化才能得到二值边缘；
 *          Canny 内部包含非极大值抑制 + 双阈值 + 滞后连接，是三者中唯一边缘「细且连通」的。
 *          输入为视频时逐帧处理并写出三段视频，输入为图片时输出拼接对比图。
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
 * @brief  对单帧做三种边缘检测。
 * @param  bgr      输入彩色帧。
 * @param  lowTh    双阈值下限。
 * @param  highTh   双阈值上限。
 * @param  grayOut  输出：灰度图。
 * @param  sobelOut 输出：Sobel 梯度幅值。
 * @param  lapOut   输出：Laplacian 响应（取绝对值）。
 * @param  cannyOut 输出：Canny 二值边缘。
 */
void detectEdges(const cv::Mat& bgr, double lowTh, double highTh, cv::Mat& grayOut,
                 cv::Mat& sobelOut, cv::Mat& lapOut, cv::Mat& cannyOut)
{
    // 步骤 1：灰度转换。
    cv::cvtColor(bgr, grayOut, cv::COLOR_BGR2GRAY);

    // 步骤 2：高斯降噪。差分算子对噪声极敏感，必须先平滑。
    cv::Mat blurred;
    cv::GaussianBlur(grayOut, blurred, cv::Size(5, 5), 0);

    // 步骤 3：计算梯度幅值（Sobel）与二阶导（Laplacian）。
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(blurred, gx, CV_64F, 1, 0, 3);
    cv::Sobel(blurred, gy, CV_64F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    cv::convertScaleAbs(magnitude, sobelOut);

    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_64F, 3);
    cv::convertScaleAbs(laplacian, lapOut);

    // 步骤 4：双阈值处理（Canny 内部完成 NMS 与滞后连接）。
    cv::Canny(blurred, cannyOut, lowTh, highTh);
}

}  // namespace

/**
 * @brief  主流程：图片模式输出拼接图，视频模式逐帧写三段视频。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 图片或视频路径，缺省读取 assets/scene.png }"
        "{o output       |      | 输出目录或图像路径，缺省写入 <目录>/output/ }"
        "{l lowThresh    | 100  | Canny 双阈值下限 }"
        "{u highThresh   | 200  | Canny 双阈值上限 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("边缘检测演示: Sobel / Laplacian / Canny\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("scene.png");
    }
    const double lowTh = parser.get<double>("lowThresh");
    const double highTh = parser.get<double>("highThresh");

    std::string outDir = parser.get<std::string>("output");
    if (outDir.empty()) {
        outDir = outputDirFor(inputPath);
    }
    cv::utils::fs::createDirectories(outDir);

    // 先按图片尝试；读不到再按视频处理。
    const cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (!image.empty()) {
        cv::Mat gray;
        cv::Mat sobel;
        cv::Mat laplacian;
        cv::Mat canny;
        detectEdges(image, lowTh, highTh, gray, sobel, laplacian, canny);

        const std::vector<cv::Mat> panels = {
            fitPanel(gray, "gray input"),
            fitPanel(sobel, "sobel magnitude"),
            fitPanel(laplacian, "laplacian"),
            fitPanel(canny, "canny (" + std::to_string(static_cast<int>(lowTh)) + "," +
                                std::to_string(static_cast<int>(highTh)) + ")")};
        const cv::Mat result = montage(panels, 2);

        const std::string outputPath = outDir + "/edge_detection.png";
        if (!cv::imwrite(outputPath, result)) {
            std::cerr << "无法写入输出图像: " << outputPath << std::endl;
            return -1;
        }
        std::cout << "图片模式: " << image.cols << "x" << image.rows
                  << ", 结果已保存至 " << outputPath << std::endl;
        if (parser.get<int>("display") != 0) {
            cv::imshow("edge detection", result);
            cv::waitKey(0);
        }
        return 0;
    }

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "无法打开图片或视频: " << inputPath << std::endl;
        return -1;
    }
    const double fps = capture.get(cv::CAP_PROP_FPS) > 0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    cv::VideoWriter sobelWriter(outDir + "/sobel.mp4", fourcc, fps, cv::Size(width, height), false);
    cv::VideoWriter lapWriter(outDir + "/laplacian.mp4", fourcc, fps, cv::Size(width, height), false);
    cv::VideoWriter cannyWriter(outDir + "/canny.mp4", fourcc, fps, cv::Size(width, height), false);
    if (!sobelWriter.isOpened() || !lapWriter.isOpened() || !cannyWriter.isOpened()) {
        std::cerr << "无法创建输出视频" << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frames = 0;
    while (capture.read(frame)) {
        cv::Mat gray;
        cv::Mat sobel;
        cv::Mat laplacian;
        cv::Mat canny;
        detectEdges(frame, lowTh, highTh, gray, sobel, laplacian, canny);
        sobelWriter.write(sobel);
        lapWriter.write(laplacian);
        cannyWriter.write(canny);
        ++frames;
        if (parser.get<int>("display") != 0 && frames % 5 == 0) {
            cv::imshow("canny", canny);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }
    std::cout << "视频模式: 处理 " << frames << " 帧, 已写出 sobel/laplacian/canny.mp4 到 " << outDir
              << std::endl;
    return 0;
}
