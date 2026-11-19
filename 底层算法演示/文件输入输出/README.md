# 文件输入输出

## 需求
读写图像视频文件，XML/YAML存储数据

## 作用

- **结果落盘**：把处理结果存成 PNG 或 JPG 供引用。
- **参数持久化**：把标定参数与配置写进 XML 或 YAML。
- **批处理流水线**：统一读写路径与编码质量。
- **中间结果缓存**：把耗时中间量存下来供下次复用。

适用边界：JPEG 有损，参数与尺寸类结果应存 PNG 或结构化文件。

## 核心函数
cv2.imread(), cv2.imwrite(), cv2.FileStorage()

## 关键步骤
1. 读取图像视频
2. 保存结果
3. XML/YAML读写
4. 序列化参数

## 输入
文件路径

## 输出
处理结果文件

## 素材
`assets/`：`image.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/file_io

# Python
cd Python && pip install -r requirements.txt && python file_io.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出目录、`-q/--quality` JPEG 质量、`-d/--display` 弹窗。
会在输出目录生成 `file_io_copy.png`、`file_io_copy.jpg`、`params.xml`、`params.yml`；
实测 quality=90 时 JPEG 读回平均差异 3.64~5.09（PNG 路径无损），XML/YAML 中的 Mat 可完整读回。
