/**
 * @file    background_subtraction.cpp
 * @brief   背景减除演示：从视频序列中分离前景运动物体。
 * @details 流程：读取视频 -> 创建背景减除器 -> 逐帧计算前景掩码 -> 形态学优化 -> 提取目标 -> 输出/显示。
 * @note    依赖：OpenCV 4.x（core / imgproc / videoio / highgui / video）
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

/// 目标最小面积（像素）。小于该面积的轮廓视为噪声，可被过滤。
static const double kMinTargetArea = 500.0;

/**
 * @brief  在若干候选位置中查找素材，便于从不同工作目录运行。
 * @param  fileName 素材文件名。
 * @return 第一个存在的路径；都不存在时返回候选列表的第一项。
 */
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

/**
 * @brief  由素材路径推导输出目录：<demo>/assets/x.mp4 -> <demo>/output。
 * @param  inputPath 素材路径。
 * @return 输出目录路径。
 */
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

/**
 * @brief  创建背景减除器。
 * @param  method 背景减除算法名称，可选 "MOG2" 或 "KNN"。
 * @return OpenCV 背景减除器智能指针。
 * @throw  std::invalid_argument 当 method 非法时抛出。
 */
cv::Ptr<cv::BackgroundSubtractor> createSubtractor(const std::string& method)
{
    if (method == "MOG2") {
        // history=500      : 用于建立背景模型的帧数；
        // varThreshold=16  : 像素与背景模型匹配的方差阈值，越小越敏感；
        // detectShadows=true: 阴影被标记为 127，确定前景为 255。
        return cv::createBackgroundSubtractorMOG2(500, 16.0, true);
    }
    if (method == "KNN") {
        // history=500        : 用于建立背景模型的帧数；
        // dist2Threshold=400 : 样本与像素的距离平方阈值，越小越敏感；
        // detectShadows=true : 阴影被标记为 127，确定前景为 255。
        return cv::createBackgroundSubtractorKNN(500, 400.0, true);
    }
    throw std::invalid_argument("未知的背景减除方法: " + method + " (可选 MOG2 / KNN)");
}

/**
 * @brief  形态学优化前景掩码：去除噪点并填补目标内部空洞。
 * @param  mask 背景减除器输出的单通道前景掩码（取值 0 / 127 / 255）。
 * @return 优化后的二值前景掩码（取值 0 / 255）。
 */
cv::Mat refineMask(const cv::Mat& mask)
{
    cv::Mat binary;
    // 1) 去阴影：仅保留确定前景（255），丢弃阴影（127）与背景（0）。
    cv::threshold(mask, binary, 200, 255, cv::THRESH_BINARY);
    // 2) 形态学优化：开运算去孤立噪点，闭运算填补目标内部空洞。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    return binary;
}

/**
 * @brief  在掩码上提取运动目标轮廓，并过滤小面积噪声。
 * @param  mask    二值前景掩码（单通道，0 / 255）。
 * @param  minArea 目标最小面积（像素），小于该值的轮廓被忽略。
 * @return 目标轮廓集合，每个元素为一条轮廓的点集。
 */
std::vector<std::vector<cv::Point>> extractTargets(const cv::Mat& mask, double minArea)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> targets;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= minArea) {
            targets.push_back(contour);
        }
    }
    return targets;
}

/**
 * @brief  将二值掩码转为可视化图像，并用矩形框标出运动目标。
 * @param  mask    二值前景掩码（单通道，0 / 255）。
 * @param  targets 目标轮廓集合。
 * @return 三通道 BGR 可视化图像。
 */
cv::Mat drawTargets(const cv::Mat& mask, const std::vector<std::vector<cv::Point>>& targets)
{
    cv::Mat vis;
    cv::cvtColor(mask, vis, cv::COLOR_GRAY2BGR);
    for (const auto& contour : targets) {
        cv::rectangle(vis, cv::boundingRect(contour), cv::Scalar(0, 0, 255), 2);
    }
    return vis;
}

/**
 * @brief  主流程：解析参数、逐帧背景减除、保存掩码并可选显示。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    // 注意：OpenCV 4.6 的 CommandLineParser 只接受 --key=value / -k=value，
    // 写成 --input xxx 会把 xxx 当成位置参数而忽略，因此这里不使用 @位置参数。
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入视频路径, 缺省读取 assets/car_passing.mp4 }"
        "{m method       | MOG2 | 背景减除方法: MOG2 或 KNN }"
        "{o output       |      | 前景掩码输出视频路径, 缺省写入 <目录>/output/mask_<方法>.mp4 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("背景减除演示：从视频序列中分离前景运动物体\n"
                 "参数请使用 --key=value 形式，例如 --method=MOG2 --display=1");
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
        inputPath = locateAsset("car_passing.mp4");
    }
    const std::string method = parser.get<std::string>("method");
    const bool display = parser.get<int>("display") != 0;

    // 步骤 1：打开视频序列
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << inputPath << std::endl;
        return -1;
    }

    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 输出路径缺省为：<目录>/output/mask_<方法>.mp4
    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/mask_" + method + ".mp4";
    }

    cv::VideoWriter writer;
    writer.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                cv::Size(width, height), false);
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频: " << outputPath << std::endl;
        return -1;
    }

    // 步骤 2：创建背景减除器
    cv::Ptr<cv::BackgroundSubtractor> subtractor = createSubtractor(method);

    cv::Mat frame;
    cv::Mat mask;
    int frameCount = 0;
    while (cap.read(frame)) {
        // 步骤 3：逐帧获取前景掩码
        subtractor->apply(frame, mask);

        // 步骤 4：形态学优化
        const cv::Mat binary = refineMask(mask);
        // const cv::Mat binary = mask;

        // 步骤 5：提取目标
        const std::vector<std::vector<cv::Point>> targets = extractTargets(binary, kMinTargetArea);

        writer.write(binary);
        if (display) {
            cv::imshow("input", frame);
            cv::imshow("foreground mask", drawTargets(binary, targets));
            if (cv::waitKey(30) == 27) {  // ESC 退出
                break;
            }
        }
        ++frameCount;
    }

    std::cout << "处理完成: " << frameCount << " 帧, 前景掩码已保存至 " << outputPath << std::endl;
    return 0;
}
