/**
 * @file    geometric_drawing.cpp
 * @brief   几何绘制演示：在画布上绘制基本几何形状、样式与文字标注。
 * @details 流程：创建画布 -> 绘制几何形状 -> 设置样式(线宽/线型/填充/亚像素) -> 标注文字 -> 保存/显示。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          cv::putText 只支持 ASCII（Hershey 矢量字体），中文需借助 freetype 模块。
 */

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

/// 常用配色（BGR 顺序）。
const cv::Scalar kRed(60, 60, 220);
const cv::Scalar kBlue(220, 120, 40);
const cv::Scalar kGreen(60, 160, 60);
const cv::Scalar kMagenta(200, 60, 200);
const cv::Scalar kOrange(0, 150, 255);
const cv::Scalar kGray(120, 120, 120);

/**
 * @brief  在若干候选位置中查找素材，便于从不同工作目录运行。
 * @param  fileName 素材文件名，如 canvas.png。
 * @return 第一个存在的路径；都不存在时返回首选候选。
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
 * @brief  由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。
 * @param  inputPath 素材路径。
 * @return 输出目录路径（尚未创建）。
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
 * @brief  载入画布；素材缺失时创建纯白画布。
 * @param  path   画布素材路径。
 * @param  width  空白画布宽度。
 * @param  height 空白画布高度。
 * @return 三通道 BGR 画布。
 */
cv::Mat loadOrCreateCanvas(const std::string& path, int width, int height)
{
    if (!path.empty()) {
        const cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
        if (!src.empty()) {
            return src.clone();
        }
        std::cerr << "无法读取画布素材，改用空白画布: " << path << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
}

/**
 * @brief  绘制空心几何形状：不同线宽、矩形、圆、椭圆、折线与箭头。
 * @param  canvas 目标画布。
 * @note   线型用 cv::LINE_AA 抗锯齿；thickness 为 -1 表示填充。
 */
void drawOutlines(cv::Mat& canvas)
{
    // 1) 同一位置的直线：线宽 1 / 3 / 7 对比，粗线更醒目但会吃掉细节。
    cv::line(canvas, cv::Point(60, 90), cv::Point(460, 90), kRed, 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(60, 150), cv::Point(460, 150), kRed, 3, cv::LINE_AA);
    cv::line(canvas, cv::Point(60, 220), cv::Point(460, 220), kRed, 7, cv::LINE_AA);
    cv::putText(canvas, "thickness 1 / 3 / 7", cv::Point(60, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, kGray, 1, cv::LINE_AA);

    // 2) 空心矩形与圆。
    cv::rectangle(canvas, cv::Rect(60, 300, 240, 150), kBlue, 2, cv::LINE_AA);
    cv::circle(canvas, cv::Point(400, 375), 78, kGreen, 3, cv::LINE_AA);

    // 3) 椭圆：angle 控制长轴倾角，区间 0~360 表示画完整椭圆。
    cv::ellipse(canvas, cv::Point(250, 600), cv::Size(180, 80), 20.0, 0.0, 360.0,
                kMagenta, 2, cv::LINE_AA);

    // 4) 带箭头的直线。
    cv::arrowedLine(canvas, cv::Point(60, 740), cv::Point(470, 800), kOrange, 2,
                    cv::LINE_AA, 0, 0.05);
}

/**
 * @brief  绘制填充图形、多边形与亚像素线条。
 * @param  canvas 目标画布。
 */
void drawFilled(cv::Mat& canvas)
{
    // 1) 填充矩形与填充圆：thickness = -1。
    cv::rectangle(canvas, cv::Rect(560, 80, 180, 150), kBlue, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, cv::Point(870, 155), 75, kGreen, cv::FILLED, cv::LINE_AA);

    // 2) fillPoly：用点集围成的任意多边形。
    const std::vector<cv::Point> triangle = {
        cv::Point(640, 300), cv::Point(560, 450), cv::Point(720, 450)};
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{triangle}, kMagenta,
                 cv::LINE_AA);

    // 3) polylines：一次画多条折线（这里画折线与拐点标记）。
    const std::vector<cv::Point> zigzag = {cv::Point(780, 300), cv::Point(850, 380),
                                           cv::Point(920, 310), cv::Point(960, 450)};
    cv::polylines(canvas, std::vector<std::vector<cv::Point>>{zigzag}, false, kRed, 2,
                  cv::LINE_AA);
    for (const auto& pt : zigzag) {
        cv::circle(canvas, pt, 5, kOrange, cv::FILLED, cv::LINE_AA);
    }

    // 4) 亚像素坐标：坐标左移 shift 位做定点数运算，可画在 .5 像素位置上。
    const int shift = 4;  // 定点小数位，坐标值需乘以 2^shift
    cv::line(canvas, cv::Point(560 << shift, 534 << shift),
             cv::Point(960 << shift, 534 << shift), kGray, 1, cv::LINE_AA, shift);
    cv::putText(canvas, "shift=4 (sub-pixel)", cv::Point(560, 580),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, kGray, 1, cv::LINE_AA);
}

/**
 * @brief  绘制文字：字体、字号、粗细、描边背景与亚像素定位。
 * @param  canvas 目标画布。
 * @note   中文无法用 cv::putText 渲染，只能走 freetype 模块的 cv::freetype::putText。
 */
void drawText(cv::Mat& canvas)
{
    cv::putText(canvas, "OpenCV Drawing", cv::Point(60, 900),
                cv::FONT_HERSHEY_SIMPLEX, 1.6, kBlue, 3, cv::LINE_AA);
    cv::putText(canvas, "FONT_HERSHEY_DUPLEX", cv::Point(60, 950),
                cv::FONT_HERSHEY_DUPLEX, 1.0, kMagenta, 2, cv::LINE_AA);
    cv::putText(canvas, "FONT_ITALIC", cv::Point(60, 995),
                cv::FONT_ITALIC, 1.0, kGreen, 2, cv::LINE_AA);

    // 反色标签：先铺实心矩形再写白字，是最常用的标注样式。
    const std::string label = "annotated";
    int baseline = 0;
    const cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    const cv::Rect box(760, 880, textSize.width + 20, textSize.height + baseline + 16);
    cv::rectangle(canvas, box, kBlue, cv::FILLED, cv::LINE_AA);
    cv::putText(canvas, label, cv::Point(box.x + 10, box.y + textSize.height + 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

}  // namespace

/**
 * @brief  主流程：准备画布、依次绘制三类内容、保存结果。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    // OpenCV 4.6 的 CommandLineParser 只接受 --key=value 形式。
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 画布素材路径，缺省读取 assets/canvas.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/geometric_drawing.png }"
        "{w width        | 1024 | 空白画布宽度 }"
        "{h height       | 1024 | 空白画布高度 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("几何绘制演示: 在画布上绘制形状、样式与文字\n参数请使用 --key=value 形式");
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

    // 步骤 1：创建画布
    cv::Mat canvas = loadOrCreateCanvas(inputPath, parser.get<int>("width"),
                                        parser.get<int>("height"));
    if (canvas.empty()) {
        std::cerr << "画布创建失败" << std::endl;
        return -1;
    }

    // 步骤 2：绘制几何形状
    drawOutlines(canvas);
    drawFilled(canvas);

    // 步骤 3：设置样式并标注文字
    drawText(canvas);

    // 步骤 4：保存 / 显示结果
    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/geometric_drawing.png";
    }
    if (!cv::imwrite(outputPath, canvas)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "画布尺寸: " << canvas.cols << "x" << canvas.rows
              << ", 结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("geometric drawing", canvas);
        cv::waitKey(0);
    }
    return 0;
}
