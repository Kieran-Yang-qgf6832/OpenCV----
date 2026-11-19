/**
 * @file    planar_object_detection.cpp
 * @brief   平面物体检测演示：特征匹配 + 单应估计 + 姿态分解，在场景中定位平面物体。
 * @details 流程：特征检测匹配 -> 计算单应性 -> 分解姿态 -> 绘制检测框。
 * @note    依赖：OpenCV 4.x（core / features2d / calib3d / imgproc / imgcodecs / highgui）
 *          decomposeHomographyMat 需要相机内参；这里没有标定数据，采用「焦距 ≈ 1.2×长边、
 *          主点在画面中心」的虚拟内参，因此分解出的平移量只有相对意义，旋转较可靠。
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
 * @brief  主流程：匹配、估计 H、分解姿态并绘制检测框。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{s scene        |      | 场景图，缺省读取 assets/scene.png }"
        "{t template     |      | 模板图，缺省读取 assets/template.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/planar_object_detection.png }"
        "{n features     | 2000 | ORB 特征点数上限 }"
        "{r ratio        | 0.75 | Lowe 比率阈值 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("平面物体检测演示: 特征匹配 + 单应估计 + 姿态分解\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string scenePath = parser.get<std::string>("scene");
    std::string templatePath = parser.get<std::string>("template");
    if (scenePath.empty()) {
        scenePath = locateAsset("scene.png");
    }
    if (templatePath.empty()) {
        templatePath = locateAsset("template.png");
    }

    const cv::Mat sceneImage = cv::imread(scenePath, cv::IMREAD_COLOR);
    const cv::Mat templateImage = cv::imread(templatePath, cv::IMREAD_COLOR);
    if (sceneImage.empty() || templateImage.empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return -1;
    }

    // 步骤 1：特征检测匹配。
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(std::max(100, parser.get<int>("features")));
    std::vector<cv::KeyPoint> kpTemplate;
    std::vector<cv::KeyPoint> kpScene;
    cv::Mat descTemplate;
    cv::Mat descScene;
    orb->detectAndCompute(templateImage, cv::noArray(), kpTemplate, descTemplate);
    orb->detectAndCompute(sceneImage, cv::noArray(), kpScene, descScene);
    if (descTemplate.empty() || descScene.empty()) {
        std::cerr << "特征提取失败" << std::endl;
        return -1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descTemplate, descScene, knn, 2);
    std::vector<cv::DMatch> good;
    const double ratio = parser.get<double>("ratio");
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 4) {
        std::cerr << "有效匹配不足 4 组" << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> ptsTemplate;
    std::vector<cv::Point2f> ptsScene;
    for (const auto& m : good) {
        ptsTemplate.push_back(kpTemplate[m.queryIdx].pt);
        ptsScene.push_back(kpScene[m.trainIdx].pt);
    }

    // 步骤 2：计算单应性。RANSAC 剔除错误匹配。
    cv::Mat inlierMask;
    cv::Mat homography = cv::findHomography(ptsTemplate, ptsScene, cv::RANSAC, 5.0, inlierMask);
    if (homography.empty()) {
        std::cerr << "单应估计失败" << std::endl;
        return -1;
    }
    const int inliers = cv::countNonZero(inlierMask);

    // 步骤 3：分解姿态。把模板的四个角变换到场景中，顺便得到检测框。
    const std::vector<cv::Point2f> corners = {
        {0.0f, 0.0f},
        {static_cast<float>(templateImage.cols), 0.0f},
        {static_cast<float>(templateImage.cols), static_cast<float>(templateImage.rows)},
        {0.0f, static_cast<float>(templateImage.rows)}};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(corners, projected, homography);

    // 虚拟内参：无标定数据时的常用近似，平移量只有相对意义。
    const double focal = 1.2 * std::max(sceneImage.cols, sceneImage.rows);
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3)
        << focal, 0.0, sceneImage.cols / 2.0, 0.0, focal, sceneImage.rows / 2.0, 0.0, 0.0, 1.0);
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    std::vector<cv::Mat> normals;
    const int solutions = cv::decomposeHomographyMat(homography.clone(), cameraMatrix,
                                                     rotations, translations, normals);

    std::cout << "匹配数: " << good.size() << ", 内点: " << inliers << " ("
              << std::fixed << std::setprecision(1) << 100.0 * inliers / good.size() << "%)\n";
    std::cout << "姿态分解解数: " << solutions << "\n";
    if (!rotations.empty()) {
        cv::Mat rvec;
        cv::Rodrigues(rotations[0], rvec);
        std::cout << "  第 1 组解 旋转向量: [" << std::setprecision(4) << rvec.at<double>(0)
                  << ", " << rvec.at<double>(1) << ", " << rvec.at<double>(2) << "] rad\n";
        std::cout << "  第 1 组解 平移向量: [" << translations[0].at<double>(0) << ", "
                  << translations[0].at<double>(1) << ", " << translations[0].at<double>(2)
                  << "]\n";
        std::cout << "  注: 使用虚拟内参(f=" << focal << ")，平移量仅有相对意义\n";
    }

    // 步骤 4：绘制检测框。
    cv::Mat sceneView = sceneImage.clone();
    std::vector<cv::Point> polygon;
    for (const auto& pt : projected) {
        polygon.emplace_back(cvRound(pt.x), cvRound(pt.y));
    }
    cv::polylines(sceneView, std::vector<std::vector<cv::Point>>{polygon}, true,
                  cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    for (const auto& pt : polygon) {
        cv::circle(sceneView, pt, 6, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }

    // 匹配可视化与矫正结果。
    cv::Mat matchesView;
    cv::drawMatches(templateImage, kpTemplate, sceneImage, kpScene, good, matchesView,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(inlierMask.begin<uchar>(), inlierMask.end<uchar>()),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::Mat rectified;
    cv::warpPerspective(sceneImage, rectified, homography.inv(), templateImage.size());

    const std::vector<cv::Mat> panels = {
        fitPanel(templateImage, "template"),
        fitPanel(sceneView, "detected: " + std::to_string(inliers) + " inliers"),
        fitPanel(matchesView, "inlier matches"),
        fitPanel(rectified, "rectified object")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(scenePath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/planar_object_detection.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("planar object detection", result);
        cv::waitKey(0);
    }
    return 0;
}
