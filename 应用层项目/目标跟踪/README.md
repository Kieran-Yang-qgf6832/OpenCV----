# 目标跟踪

## 需求
给定首帧的初始框，跟踪后续帧中的运动目标，并量化跟踪误差。

## 作用

- **监控跟拍**：锁定一个人/车持续记录轨迹。
- **无人机跟随**：单目标跟踪是跟随控制的外环。
- **计数与测速**：轨迹长度换算速度。

适用边界：遮挡、快速形变、出画后无法自动找回；初始框必须包含完整目标。

## 核心函数
`cv2.TrackerCSRT_create()`、`tracker.init()`、`tracker.update()`

## 关键步骤
1. 读首帧与 `init_box.txt` 里的初始框，初始化 CSRT 跟踪器
2. 逐帧 `update` 得到新框，把轨迹画成渐变折线
3. 目标是很纯的红色，用颜色分割质心作为独立真值，逐帧算中心误差
4. 输出跟踪视频、误差曲线与统计

## 输入
`target.mp4`（100 帧，目标从左向右横移）+ `init_box.txt`

## 输出
跟踪视频、关键帧、误差曲线

## 素材
`assets/target.mp4`：走廊里一个红色苹果横向移动，背景静止；
`assets/init_box.txt`：`70,396,109,114`。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --tracker KCF      # 换算法对比
```
依赖 OpenCV（含 contrib，tracking 在 contrib 里）+ NumPy。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/tracked.mp4` | 绿框 + 橙色轨迹 + 实时误差 |
| `output/track_frame.png` | 中间帧 |
| `output/track_error.png` | 中心误差曲线 |

实测：CSRT 跟踪 100 帧全部成功，中心误差 **mean 1.94 px / max 3.30 px**，
超过 10 px 的帧占比 **0%**。

## 实现要点
- **用独立手段当真值**：颜色质心与跟踪器完全独立，误差数字才有意义。
- **CSRT vs KCF**：CSRT 精度更高但更慢；KCF 快，目标缩放时容易漂。
- **按需限制轨迹长度**：只画最近 40 帧，长视频里轨迹不会糊成一团。
