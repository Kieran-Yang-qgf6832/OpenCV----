/**
 * @file    file_io.cpp
 * @brief   文件输入输出演示：图像读写、编码参数控制与 XML/YAML 参数持久化。
 * @details 流程：读取图像 -> 保存不同格式与质量 -> XML/YAML 读写 -> 序列化参数。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          FileStorage 会根据扩展名自动选择 XML 或 YAML；写入的 Mat 会带尺寸与类型，
 *          读回时用 Mat 名节点即可还原，是保存标定参数/模型参数的常用手段。
 */

#include <algorithm>
#include <ctime>
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

/** @brief 把 FileStorage 中的处理参数写成两份（XML 与 YAML）。 */
void writeParams(const std::string& xmlPath, const std::string& yamlPath,
                 const cv::Mat& sample, int quality)
{
    for (const std::string& path : {xmlPath, yamlPath}) {
        cv::FileStorage storage(path, cv::FileStorage::WRITE);
        if (!storage.isOpened()) {
            continue;
        }
        const std::time_t now = std::time(nullptr);
        storage << "created" << cv::String(std::ctime(&now));
        storage << "jpeg_quality" << quality;
        storage << "resize_scale" << 0.5;
        storage << "sample_region" << sample;
        storage.release();
    }
}

}  // namespace

/**
 * @brief  主流程：读写图像与参数文件。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/image.png }"
        "{o output       |      | 输出目录，缺省写入 <目录>/output }"
        "{q quality      | 90   | JPEG 压缩质量 0~100 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("文件输入输出演示: imread/imwrite/FileStorage\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("image.png");
    }
    std::string outDir = parser.get<std::string>("output");
    if (outDir.empty()) {
        outDir = outputDirFor(inputPath);
    }
    cv::utils::fs::createDirectories(outDir);
    const int quality = std::max(1, std::min(100, parser.get<int>("quality")));

    // 步骤 1：读取图像。
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    std::cout << "读取: " << inputPath << " (" << image.cols << "x" << image.rows << ", "
              << image.channels() << " 通道)\n";

    // 步骤 2：保存结果。PNG 无损，JPEG 有损且受 quality 控制。
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    const std::string pngPath = outDir + "/file_io_copy.png";
    const std::string jpgPath = outDir + "/file_io_copy.jpg";
    cv::imwrite(pngPath, image);
    const std::vector<int> jpegParams = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imwrite(jpgPath, image, jpegParams);

    // 重新读回 JPEG，量化有损压缩带来的差异。
    const cv::Mat reloaded = cv::imread(jpgPath, cv::IMREAD_COLOR);
    double meanDiff = 0.0;
    if (!reloaded.empty()) {
        cv::Mat diff;
        cv::absdiff(image, reloaded, diff);
        meanDiff = cv::mean(diff)[0];
    }
    std::cout << "写出: " << pngPath << "\n";
    std::cout << "写出: " << jpgPath << " (quality=" << quality << ", 读回平均差异 "
              << std::fixed << std::setprecision(2) << meanDiff << ")\n";

    // 步骤 3~4：XML / YAML 读写与序列化。
    cv::Mat sample = image(cv::Rect(0, 0, 64, 64)).clone();
    const std::string xmlPath = outDir + "/params.xml";
    const std::string yamlPath = outDir + "/params.yml";
    writeParams(xmlPath, yamlPath, sample, quality);

    cv::FileStorage reader(xmlPath, cv::FileStorage::READ);
    if (reader.isOpened()) {
        int readQuality = 0;
        double readScale = 0.0;
        cv::Mat readRegion;
        reader["jpeg_quality"] >> readQuality;
        reader["resize_scale"] >> readScale;
        reader["sample_region"] >> readRegion;
        reader.release();
        std::cout << "从 " << xmlPath << " 读回: jpeg_quality=" << readQuality
                  << ", resize_scale=" << readScale << ", sample_region=" << readRegion.cols
                  << "x" << readRegion.rows << "\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(image, "original"),
        fitPanel(resized, "resized 50%"),
        fitPanel(gray, "gray"),
        fitPanel(sample, "region 64x64")};
    const cv::Mat result = montage(panels, 2);

    const std::string outputPath = outDir + "/file_io.png";
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("file io", result);
        cv::waitKey(0);
    }
    return 0;
}
