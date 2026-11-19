# FLANN匹配器

## 需求
使用FLANN进行高效特征匹配

## 作用

- **大特征集匹配**：描述子上万时替代暴力匹配。
- **图像检索**：在海量特征库里做近邻查找。
- **三维重建**：为大量同名点快速找候选。
- **实时系统**：在保证匹配率的前提下压低耗时。

适用边界：数据量小时索引构建开销会让它比暴力匹配更慢。

## 核心函数
cv2.FlannBasedMatcher()

## 关键步骤
1. 选择索引算法
2. 设置搜索参数
3. kNN匹配
4. 比率测试筛选

## 输入
两组描述子

## 输出
高质量匹配

## 素材
`assets/`：`scene.png`、`scene_rotated.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/flann_matching

# Python
cd Python && pip install -r requirements.txt && python flann_matching.py
```
参数：`-a/--imageA`、`-b/--imageB`、`-o/--output` 输出路径、`-r/--ratio` 比率、`-t/--trees` KD 树数、`-c/--checks` 搜索叶节点数、`-d/--display` 弹窗。
实测（SIFT 1500 描述子）：FLANN 773 组好匹配 / 48 ms，BFMatcher 776 组 / 6 ms。
**这个尺寸下 FLANN 反而更慢**——数据量小时近似索引的构建开销占主导；FLANN 的优势要到数万级描述子才体现。
