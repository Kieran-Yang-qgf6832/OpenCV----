# 视频质量评估

## 需求
计算视频的PSNR和SSIM质量指标

## 核心函数
cv2.PSNR(), cv2.quality.QualitySSIM_create()

## 关键步骤
1. 读取参考和测试视频
2. 逐帧计算PSNR
3. 逐帧计算SSIM
4. 统计平均质量

## 输入
参考视频+测试视频

## 输出
PSNR和SSIM值
