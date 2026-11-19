/**
 * @file    random_and_text.cpp
 * @brief   随机数与文本演示：用 cv::RNG / randu / randn 生成随机数据并绘制随机图形与文字。
 * @details 流程：创建 RNG -> 生成随机数矩阵 -> 绘制随机文本 -> 随机图形绘制。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          cv::RNG 是固定种子的线性同余发生器，给定种子结果完全可复现；
 *          randu 产生均匀分布，randn 产生正态分布，二者都直接写入 Mat。
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

/// 随机文本使用的字符表（cv::putText 只支持 ASCII）。
const std::string kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

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

/** @brief 生成指定长度的随机字符串。 */
std::string randomString(cv::RNG& rng, int length)
{
    std::string text;
    for (int i = 0; i < length; ++i) {
        text += kAlphabet[static_cast<size_t>(rng.uniform(0, static_cast<int>(kAlphabet.size())))];
    }
    return text;
}

/** @brief 生成一个随机 BGR 颜色，并保证足够亮以便看清。 */
cv::Scalar randomColor(cv::RNG& rng)
{
    return cv::Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
}

}  // namespace

/**
 * @brief  主流程：随机数矩阵 + 随机图形 + 随机文本。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 画布素材，缺省读取 assets/canvas.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/random_and_text.png }"
        "{s seed         | 20261119 | 随机种子，相同种子结果完全一致 }"
        "{n count        | 40   | 随机图形与文本的数量 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("随机数与文本演示: RNG / randu / randn / putText\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("canvas.png");
    }
    cv::Mat canvas = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (canvas.empty()) {
        canvas = cv::Mat(1024, 1024, CV_8UC3, cv::Scalar(255, 255, 255));
    }

    // 步骤 1：创建 RNG 对象。种子相同则整个序列完全可复现。
    cv::RNG rng(parser.get<int>("seed"));
    const int count = std::max(1, parser.get<int>("count"));

    // 步骤 2：生成随机数矩阵。
    cv::Mat uniformNoise(256, 256, CV_8U);
    cv::randu(uniformNoise, cv::Scalar(0), cv::Scalar(256));
    cv::Mat gaussianNoise(256, 256, CV_32F);
    cv::randn(gaussianNoise, cv::Scalar(128), cv::Scalar(40));
    cv::Mat gaussianView;
    gaussianNoise.convertTo(gaussianView, CV_8U);

    cv::Scalar uniformMean;
    cv::Scalar uniformStd;
    cv::Scalar gaussMean;
    cv::Scalar gaussStd;
    cv::meanStdDev(uniformNoise, uniformMean, uniformStd);
    cv::meanStdDev(gaussianNoise, gaussMean, gaussStd);
    std::cout << "均匀分布 randu(0,256): mean=" << std::fixed << std::setprecision(1)
              << uniformMean[0] << ", std=" << uniformStd[0] << "\n";
    std::cout << "正态分布 randn(128,40): mean=" << gaussMean[0] << ", std=" << gaussStd[0] << "\n";

    // 步骤 3：绘制随机文本。
    cv::Mat textView = canvas.clone();
    for (int i = 0; i < count; ++i) {
        const cv::Point origin(rng.uniform(20, std::max(21, textView.cols - 260)),
                               rng.uniform(40, std::max(41, textView.rows - 40)));
        const double fontScale = rng.uniform(0.6, 1.6);
        cv::putText(textView, randomString(rng, rng.uniform(3, 10)), origin,
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, randomColor(rng),
                    rng.uniform(1, 3), cv::LINE_AA);
    }
    cv::putText(textView,
                "seed " + std::to_string(parser.get<int>("seed")) + " text " +
                    std::to_string(count),
                cv::Point(20, textView.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    // 步骤 4：随机图形绘制。
    cv::Mat shapeView = canvas.clone();
    for (int i = 0; i < count; ++i) {
        const cv::Point p1(rng.uniform(0, shapeView.cols), rng.uniform(0, shapeView.rows));
        const cv::Point p2(rng.uniform(0, shapeView.cols), rng.uniform(0, shapeView.rows));
        const cv::Scalar color = randomColor(rng);
        const int thickness = rng.uniform(-1, 5);  // -1 表示填充
        switch (rng.uniform(0, 4)) {
            case 0:
                cv::line(shapeView, p1, p2, color, std::max(1, thickness), cv::LINE_AA);
                break;
            case 1:
                cv::circle(shapeView, p1, rng.uniform(10, 80), color,
                           thickness < 0 ? cv::FILLED : thickness, cv::LINE_AA);
                break;
            case 2:
                cv::rectangle(shapeView, p1, p2, color,
                              thickness < 0 ? cv::FILLED : thickness, cv::LINE_AA);
                break;
            default: {
                std::vector<cv::Point> triangle = {
                    p1, p2, cv::Point(rng.uniform(0, shapeView.cols), rng.uniform(0, shapeView.rows))};
                cv::fillPoly(shapeView, std::vector<std::vector<cv::Point>>{triangle},
                             color, cv::LINE_AA);
                break;
            }
        }
    }
    cv::putText(shapeView,
                "seed " + std::to_string(parser.get<int>("seed")) + " shapes " +
                    std::to_string(count),
                cv::Point(20, shapeView.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    const std::vector<cv::Mat> panels = {
        fitPanel(uniformNoise, "randu uniform(0,256)"),
        fitPanel(gaussianView, "randn normal(128,40)"),
        fitPanel(shapeView, "random shapes"),
        fitPanel(textView, "random text")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/random_and_text.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("random and text", result);
        cv::waitKey(0);
    }
    return 0;
}
