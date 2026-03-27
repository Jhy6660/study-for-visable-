# R1 Vision（单一项目说明文档）

> 本仓库仅保留本文件作为项目说明与交接文档。

## 1. 项目目标

R1 Vision 是一个 ROS2 Python 视觉定位节点：
- 输入：RGB、Depth、LiDAR（Livox mid360）
- 输出：抓取点（优先 `base_link`，TF 不可用时可降级 `camera_link`）

核心策略：
1. YOLO 检测目标框
2. LiDAR 点云预过滤（避免全量点云进入后续）
3. 可选“相机+雷达点云融合”
4. Frustum + 平面拟合得到侧面中心与法向
5. 重力约束 + 法向平滑 + 跟踪
6. 抓取点稳定性闸门后再发送

---

## 2. 当前已核对结论（代码对齐）

### 2.1 话题默认值（代码已对齐）
- `topic_rgb`: `/camera/camera/color/image_raw`
- `topic_depth`: `/camera/camera/aligned_depth_to_color/image_raw`
- `topic_lidar_points`: `/livox/lidar`
- `topic_imu`: `/livox/imu`
- `topic_odom`: `/Odometry`

说明：相机内参当前是硬编码，不依赖 `camera_info` 订阅。

### 2.2 主流程逻辑
- 同步回调里做检测、点云读取、预过滤、融合定位、抓取计算。
- 点云不会“全量直通”：
  - 先做雷达预过滤（range/y/z）
  - 再做 Frustum 截取与前景深度门限
  - 再做统计离群点过滤

### 2.3 Launch 启动行为
- `r1_vision.launch.py`：仅启动 r1_vision 节点。
- `r1_vision_with_fastlio.launch.py`：启动 body->base 静态 TF、camera->base 静态 TF、FAST-LIO、r1_vision、可选 RViz。

---

## 3. 关键参数（建议先从这里调）

### 3.1 同步与实时性
- `sync_slop`（默认 0.03）
- `process_every_n_frames`
- `adaptive_frame_skip`
- `target_processing_ms`

### 3.2 雷达预过滤（先控算力和噪声）
- `lidar_prefilter_enable`（默认 true）
- `lidar_min_range_m`（默认 0.2）
- `lidar_max_range_m`（默认 4.0）
- `lidar_max_abs_y_m`（默认 2.5）
- `lidar_max_abs_z_m`（默认 2.0）

### 3.3 融合与几何
- `use_pointcloud_fusion`
- `fusion_max_distance`
- `fusion_min_confidence`
- `frustum_min_points`（默认 15）
- `frustum_min_cube_points`（默认 15）
- `cube_side_length`

### 3.4 抓取稳定闸门
- `grasp_consistency_enable`
- `grasp_confidence_threshold`
- `grasp_max_jump_m`
- `grasp_max_normal_angle_deg`
- `grasp_min_confirm_frames`

---

## 4. 启动示例

### 4.1 仅视觉节点
```bash
ros2 launch r1_vision r1_vision.launch.py
```

### 4.2 FAST-LIO 集成
```bash
ros2 launch r1_vision r1_vision_with_fastlio.launch.py
```

如现场话题不同，直接在 launch 传参，例如：
```bash
ros2 launch r1_vision r1_vision.launch.py topic_lidar_points:=/your/lidar/topic topic_imu:=/your/imu/topic
```

---

## 5. 推荐排查顺序（现场）

1. 先看输入话题是否正常（RGB/Depth/LiDAR/IMU/Odom）。
2. 再看 TF 是否通（`camera_link -> base_link`）。
3. 调雷达预过滤范围，让进入主流程的点数可控。
4. 调 Frustum 与融合参数，稳定法向和中心。
5. 最后调抓取稳定闸门阈值，减少误发。

---

## 6. 版本说明

- 文档已收敛为单一 README。
- 其他历史状态文档已移除，避免维护分叉。
