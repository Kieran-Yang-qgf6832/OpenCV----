/**
 * @file    periodic_noise.cpp
 * @brief   周期性噪声演示：在频域定位噪声峰值并用陷波滤波器去除。
 * @details 流程：傅里叶变换 -> 定位噪声峰值 -> 构造陷波滤波器 -> 逆变换复原。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          周期性噪声在频域表现为成对出现的孤立亮点（关于零频对称），
 *          用陷波掩膜把它们和对称点一起挖掉即可，比空域滤波干净得多。
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

/** @brief 四个象限对角互换，把零频搬到画布中心。 */
void fftShift(cv::Mat& mat)
{
    const int cx = mat.cols / 2;
    const int cy = mat.rows / 2;
    cv::Mat q0(mat, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mat, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mat, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mat, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

}  // namespace

/**
 * @brief  主流程：频谱峰值检测 + 陷波滤波。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 含噪图像，缺省读取 assets/periodic_noise.png }"
        "{r reference    |      | 干净参考图，缺省读取 assets/photo.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/periodic_noise.png }"
        "{p peaks        | 4    | 要挖掉的峰值对数量 }"
        "{n notchRadius  | 7    | 陷波半径(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("周期性噪声演示: DFT 峰值检测 + 陷波滤波\n参数请使用 --key=value 形式");
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
    std::string refPath = parser.get<std::string>("reference");
    if (inputPath.empty()) {
        inputPath = locateAsset("periodic_noise.png");
    }
    if (refPath.empty()) {
        refPath = locateAsset("photo.png");
    }

    const cv::Mat srcColor = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (srcColor.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(srcColor, gray, cv::COLOR_BGR2GRAY);
    cv::Mat srcFloat;
    gray.convertTo(srcFloat, CV_32F);

    // 步骤 1：傅里叶变换并求对数幅度谱。
    cv::Mat spectrum;
    cv::dft(srcFloat, spectrum, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> planes;
    cv::split(spectrum, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    magnitude += cv::Scalar::all(1);
    cv::log(magnitude, magnitude);
    fftShift(magnitude);

    cv::Mat spectrumView;
    cv::normalize(magnitude, spectrumView, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 步骤 2：定位噪声峰值。屏蔽中心低频区，再逐个做非极大值抑制。
    const int radius = std::max(2, parser.get<int>("notchRadius"));
    const int peakCount = std::max(1, parser.get<int>("peaks"));
    cv::Mat work = magnitude.clone();
    cv::circle(work, cv::Point(work.cols / 2, work.rows / 2), 40, cv::Scalar(0), cv::FILLED);  // 屏蔽低频主瓣，避免把图像自身结构当成噪声峰

    std::vector<cv::Point> peaks;
    const cv::Point origin(srcFloat.cols / 2, srcFloat.rows / 2);
    for (int i = 0; i < peakCount; ++i) {
        double maxValue = 0.0;
        cv::Point maxLocation;
        cv::minMaxLoc(work, nullptr, &maxValue, nullptr, &maxLocation);
        peaks.push_back(maxLocation);
        // 峰值必然成对出现，抑制时要连同对称点一起挖掉，否则会重复计入同一对。
        cv::circle(work, maxLocation, radius * 3, cv::Scalar(0), cv::FILLED);
        cv::circle(work, cv::Point(2 * origin.x - maxLocation.x, 2 * origin.y - maxLocation.y),
                   radius * 3, cv::Scalar(0), cv::FILLED);
    }

    // 步骤 3：构造陷波滤波器。峰值成对出现，对称点要一起挖掉。
    cv::Mat notch(srcFloat.size(), CV_32F, cv::Scalar(1.0f));
    const cv::Point center(srcFloat.cols / 2, srcFloat.rows / 2);
    std::cout << "输入: " << inputPath << ", 检出峰值 " << peaks.size() << " 对:\n";
    for (const auto& peak : peaks) {
        cv::circle(notch, peak, radius, cv::Scalar(0), cv::FILLED);
        const cv::Point symmetric(2 * center.x - peak.x, 2 * center.y - peak.y);
        cv::circle(notch, symmetric, radius, cv::Scalar(0), cv::FILLED);
        std::cout << "  (" << peak.x << ", " << peak.y << ") 及其对称点 (" << symmetric.x << ", "
                  << symmetric.y << ")\n";
    }

    // 步骤 4：陷波后逆变换。先反中心化掩膜，再乘到原始（零频在左上角）频谱上。
    fftShift(notch);
    planes[0] = planes[0].mul(notch);
    planes[1] = planes[1].mul(notch);
    cv::Mat filtered;
    cv::merge(planes, filtered);
    cv::Mat restored;
    cv::idft(filtered, restored, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::Mat result;
    restored.convertTo(result, CV_8U);

    // 质量评估。
    const cv::Mat reference = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
    if (!reference.empty() && reference.size() == gray.size()) {
        auto psnr = [](const cv::Mat& a, const cv::Mat& b) {
            cv::Mat diff;
            cv::absdiff(a, b, diff);
            diff.convertTo(diff, CV_32F);
            const double mse = cv::mean(diff.mul(diff))[0];
            return mse <= 0.0 ? 99.0 : 10.0 * std::log2(255.0 * 255.0 / mse) / std::log2(10.0);
        };
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  PSNR: 含噪 " << psnr(gray, reference) << " dB -> 陷波后 "
                  << psnr(result, reference) << " dB\n";
    } else {
        std::cout << "  参考图缺失，跳过 PSNR 评估\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(reference.empty() ? gray : reference, "clean reference"),
        fitPanel(gray, "periodic noise"),
        fitPanel(spectrumView, "magnitude spectrum"),
        fitPanel(result, "notch filtered")};
    const cv::Mat output = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/periodic_noise.png";
    }
    if (!cv::imwrite(outputPath, output)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("periodic noise", output);
        cv::waitKey(0);
    }
    return 0;
}
