/**
 * @file    image_inpainting.cpp
 * @brief   图像修复演示：用 TELEA 与 Navier-Stokes 两种算法修复划痕与损坏区域。
 * @details 流程：读取损坏图与掩码 -> 选择修复算法 -> 执行修复 -> 与干净参考图比较质量。
 * @note    依赖：OpenCV 4.x（core / photo / imgproc / imgcodecs / highgui）
 *          掩码必须是单通道 8 位、非零像素表示待修复区域；两种算法都靠周边像素扩散填补，
 *          TELEA 用快速行进法，速度快；NS 用流体动力学模型，纹理连续性更好但更慢。
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

/** @brief 计算两幅 8 位图像的 PSNR(dB)。 */
double psnr(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    const double mse = cv::mean(diff.mul(diff))[0];
    return mse <= 0.0 ? 99.0 : 10.0 * std::log2(255.0 * 255.0 / mse) / std::log2(10.0);
}

}  // namespace

/**
 * @brief  主流程：两种修复算法对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 损坏图像，缺省读取 assets/damaged.png }"
        "{m mask         |      | 掩码图像，缺省读取 assets/mask.png }"
        "{r reference    |      | 干净参考图，缺省读取 assets/photo.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_inpainting.png }"
        "{a radius       | 3    | 修复时参考的邻域半径 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像修复演示: TELEA / Navier-Stokes\n参数请使用 --key=value 形式");
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
    std::string maskPath = parser.get<std::string>("mask");
    std::string refPath = parser.get<std::string>("reference");
    if (inputPath.empty()) {
        inputPath = locateAsset("damaged.png");
    }
    if (maskPath.empty()) {
        maskPath = locateAsset("mask.png");
    }
    if (refPath.empty()) {
        refPath = locateAsset("photo.png");
    }

    // 步骤 1：读取损坏图与掩码。掩码必须与图像同尺寸、单通道。
    const cv::Mat damaged = cv::imread(inputPath, cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
    if (damaged.empty() || mask.empty()) {
        std::cerr << "无法读取图像或掩码" << std::endl;
        return -1;
    }
    if (mask.size() != damaged.size()) {
        cv::resize(mask, mask, damaged.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat binaryMask;
    cv::threshold(mask, binaryMask, 1, 255, cv::THRESH_BINARY);
    const int radius = std::max(1, parser.get<int>("radius"));

    // 步骤 2~3：两种算法各修复一次。
    cv::Mat telea;
    cv::Mat ns;
    cv::inpaint(damaged, binaryMask, telea, radius, cv::INPAINT_TELEA);
    cv::inpaint(damaged, binaryMask, ns, radius, cv::INPAINT_NS);

    // 步骤 4：评估质量。只在掩码区域内统计误差才有意义。
    const cv::Mat reference = cv::imread(refPath, cv::IMREAD_COLOR);
    std::cout << "输入: " << inputPath << ", 掩码像素占比: " << std::fixed
              << std::setprecision(2) << 100.0 * cv::countNonZero(binaryMask) / binaryMask.total()
              << "%\n";
    if (!reference.empty() && reference.size() == damaged.size()) {
        cv::Mat diffTelea;
        cv::Mat diffNs;
        cv::absdiff(telea, reference, diffTelea);
        cv::absdiff(ns, reference, diffNs);
        // 只在掩码区域内求均值：cv::mean 带 mask 参数时才按掩码像素数归一化。
        const cv::Scalar meanTelea = cv::mean(diffTelea, binaryMask);
        const cv::Scalar meanNs = cv::mean(diffNs, binaryMask);
        std::cout << std::setprecision(2);
        std::cout << "  全图 PSNR: TELEA " << psnr(telea, reference) << " dB, NS "
                  << psnr(ns, reference) << " dB\n";
        std::cout << "  掩码区平均绝对误差: TELEA "
                  << (meanTelea[0] + meanTelea[1] + meanTelea[2]) / 3.0 << ", NS "
                  << (meanNs[0] + meanNs[1] + meanNs[2]) / 3.0 << "\n";
    } else {
        std::cout << "  参考图缺失，跳过质量评估\n";
    }

    std::cout << "  注: 掩码面积很小（约 1%），所以全图 PSNR 差异不大，看掩码区误差更有意义\n";

    const cv::Mat cleanView = reference.empty() ? damaged : reference;
    cv::Mat teleaError;
    cv::absdiff(telea, cleanView, teleaError);
    teleaError.convertTo(teleaError, CV_8U, 4.0);  // 放大 4 倍才看得清残差

    const std::vector<cv::Mat> panels = {
        fitPanel(damaged, "damaged"),
        fitPanel(binaryMask, "mask"),
        fitPanel(telea, "TELEA"),
        fitPanel(ns, "Navier-Stokes"),
        fitPanel(cleanView, "clean reference"),
        fitPanel(teleaError, "telea error x4")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_inpainting.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image inpainting", result);
        cv::waitKey(0);
    }
    return 0;
}
