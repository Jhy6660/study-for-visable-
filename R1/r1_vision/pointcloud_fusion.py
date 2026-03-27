"""
点云融合模块 - 融合雷达和相机点云数据

雷达配置: Livox mid360
- 点云话题: /livox/lidar
- IMU话题: /livox/imu
- 坐标系: body (FAST-LIO body frame)
"""
import numpy as np
try:
    import pcl
except Exception:
    pcl = None
import torch
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


class PointCloudFusion:
    """点云融合器 - 融合雷达和相机点云"""

    def __init__(self, lidar_to_camera_transform=None):
        """
        初始化点云融合器

        Args:
            lidar_to_camera_transform: 雷达到相机的变换矩阵 (4x4)
        """
        self.lidar_to_camera_transform = lidar_to_camera_transform
        if self.lidar_to_camera_transform is None:
            # 默认变换（需要标定）
            self.lidar_to_camera_transform = np.eye(4)

    def set_transform(self, transform_matrix):
        """
        设置雷达到相机的变换矩阵

        Args:
            transform_matrix: 4x4变换矩阵
        """
        self.lidar_to_camera_transform = transform_matrix

    def transform_pointcloud(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        变换点云

        Args:
            points: 输入点云 (N x 3)
            transform_matrix: 4x4变换矩阵

        Returns:
            变换后的点云 (N x 3)
        """
        # 转换为齐次坐标
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # 应用变换
        transformed = (transform_matrix @ points_homogeneous.T).T

        # 返回前3列
        return transformed[:, :3]

    def fuse_pointclouds(
        self,
        lidar_points: np.ndarray,
        camera_points: np.ndarray,
        max_distance: float = 0.05,
        fusion_depth_weight_scale: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        融合雷达和相机点云

        Args:
            lidar_points: 雷达点云 (N x 3)
            camera_points: 相机点云 (M x 3)
            max_distance: 点云融合的最大距离阈值（米）
            fusion_depth_weight_scale: >0 时对远处点降低置信度，系数越大衰减越快（相机坐标系 Z 为深度）

        Returns:
            融合后的点云 (K x 3)
            融合后的置信度 (K,)
        """
        # 将雷达点云变换到相机坐标系
        lidar_points_camera = self.transform_pointcloud(
            lidar_points, 
            self.lidar_to_camera_transform
        )

        # 优先使用 PyTorch (GPU) 进行点云配准和融合，如果没 CUDA 就回退到 PCL
        if torch.cuda.is_available():
            try:
                return self._fuse_with_pytorch(
                    lidar_points_camera,
                    camera_points,
                    max_distance,
                    fusion_depth_weight_scale,
                )
            except Exception as e:
                print(f"PyTorch fusion failed, fallback to PCL: {e}")

        # 使用PCL进行点云配准和融合
        return self._fuse_with_pcl(
            lidar_points_camera,
            camera_points,
            max_distance,
            fusion_depth_weight_scale,
        )

    def _fuse_with_pytorch(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        max_distance: float,
        fusion_depth_weight_scale: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 PyTorch (CUDA) 批量并行计算点云融合
        """
        device = torch.device('cuda')
        
        # 显存保护：检查可用显存，不足时自动降采样
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(device).total_mem - torch.cuda.memory_allocated(device)
            max_points = int(free_mem / (4 * 4 * 2))  # 保守估计: cdist需要 4*M*N*4 bytes，取1/8作上限
            max_points = min(max_points, 30000)  # 上限3万点
            if len(points1) > max_points:
                idx1 = np.random.choice(len(points1), max_points, replace=False)
                points1 = points1[idx1]
            if len(points2) > max_points:
                idx2 = np.random.choice(len(points2), max_points, replace=False)
                points2 = points2[idx2]

        # 通用点数限制（防止极端情况）
        MAX_PER_CLOUD = 20000
        if len(points1) > MAX_PER_CLOUD:
            idx1 = np.random.choice(len(points1), MAX_PER_CLOUD, replace=False)
            points1 = points1[idx1]
        if len(points2) > MAX_PER_CLOUD:
            idx2 = np.random.choice(len(points2), MAX_PER_CLOUD, replace=False)
            points2 = points2[idx2]

        t1 = torch.tensor(points1, dtype=torch.float32, device=device)
        t2 = torch.tensor(points2, dtype=torch.float32, device=device)

        # 计算所有 point2 到 point1 的距离矩阵 (M x N)
        # 内存消耗: M*N*4 bytes. 20k x 20k = 400M * 4 = 1.6GB. (安全范围内)
        dist_matrix = torch.cdist(t2, t1)
        
        # 找到最近邻
        min_dists, min_indices = torch.min(dist_matrix, dim=1)
        
        matched_points = t1[min_indices]
        
        # 距离掩码
        valid_mask = min_dists < max_distance
        
        # 计算权重
        # nn_weight = (1.0 - dn) ** 2
        dn = torch.clamp(min_dists / max_distance, 0.0, 1.0)
        nn_weight = (1.0 - dn) ** 2
        
        # 深度加权
        z = torch.clamp(t2[:, 2], min=0.05)
        if fusion_depth_weight_scale > 1e-9:
            depth_w = 1.0 / (1.0 + fusion_depth_weight_scale * z)
        else:
            depth_w = torch.ones_like(z)
            
        weight = torch.clamp(nn_weight * depth_w, 0.0, 1.0)
        
        # 未匹配的点，权重设为 0.5，且不用匹配点更新
        weight[~valid_mask] = 0.5
        
        weight_expanded = weight.unsqueeze(1)
        
        # 融合公式
        fused_points = torch.where(
            valid_mask.unsqueeze(1),
            weight_expanded * matched_points + (1.0 - weight_expanded) * t2,
            t2
        )
        
        return fused_points.cpu().numpy(), weight.cpu().numpy()

    def _fuse_with_pcl(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        max_distance: float,
        fusion_depth_weight_scale: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PCL进行点云融合

        最近邻距离采用二次衰减；可选按深度加权（远距离点更不信任）。
        """
        if pcl is None:
            return self._fuse_with_numpy(points1, points2, max_distance, fusion_depth_weight_scale)

        # 转换为PCL点云
        cloud1 = pcl.PointCloud()
        cloud1.from_array(points1.astype(np.float32))

        cloud2 = pcl.PointCloud()
        cloud2.from_array(points2.astype(np.float32))

        # 创建KD树用于最近邻搜索
        kdtree = cloud1.make_kdtree_flann()

        # 存储融合结果
        fused_points = []
        confidences = []

        # 对点云2中的每个点，在点云1中查找最近邻
        for point in points2:
            # 转换为PCL点
            pcl_point = pcl.Point(point[0], point[1], point[2])

            # 搜索最近邻
            [indices, distances] = kdtree.nearest_k_search_for_cloud(
                pcl.PointCloud(np.array([point], dtype=np.float32)),
                1
            )

            if len(indices) > 0 and distances[0][0] < max_distance:
                matched_point = points1[indices[0][0]]
                d = float(distances[0][0])
                # 归一化距离 [0,1]，二次衰减（越近越可信）
                dn = min(d / max_distance, 1.0) if max_distance > 1e-9 else 0.0
                nn_weight = (1.0 - dn) ** 2
                # 深度加权：Z 越大（越远）略降权
                z = max(float(point[2]), 0.05)
                if fusion_depth_weight_scale > 1e-9:
                    depth_w = 1.0 / (1.0 + fusion_depth_weight_scale * z)
                else:
                    depth_w = 1.0
                weight = float(np.clip(nn_weight * depth_w, 0.0, 1.0))
                fused_point = weight * matched_point + (1.0 - weight) * point

                fused_points.append(fused_point)
                confidences.append(weight)
            else:
                fused_points.append(point)
                confidences.append(0.5)

        return np.array(fused_points), np.array(confidences)

    def _fuse_with_numpy(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        max_distance: float,
        fusion_depth_weight_scale: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points1) == 0 or len(points2) == 0:
            return points2.copy(), np.full((len(points2),), 0.5, dtype=np.float32)

        tree = cKDTree(points1)
        dists, idxs = tree.query(points2, k=1)
        valid = dists < max_distance

        dn = np.clip(dists / max(max_distance, 1e-9), 0.0, 1.0)
        nn_weight = (1.0 - dn) ** 2
        z = np.clip(points2[:, 2], 0.05, None)
        if fusion_depth_weight_scale > 1e-9:
            depth_w = 1.0 / (1.0 + fusion_depth_weight_scale * z)
        else:
            depth_w = np.ones_like(z)
        weight = np.clip(nn_weight * depth_w, 0.0, 1.0)
        weight[~valid] = 0.5

        matched = points1[idxs]
        fused = np.where(
            valid[:, None],
            weight[:, None] * matched + (1.0 - weight[:, None]) * points2,
            points2,
        )
        return fused, weight

    def filter_by_confidence(self, 
                            points: np.ndarray, 
                            confidences: np.ndarray,
                            min_confidence: float = 0.3) -> np.ndarray:
        """
        根据置信度过滤点云

        Args:
            points: 点云 (N x 3)
            confidences: 置信度 (N,)
            min_confidence: 最小置信度阈值

        Returns:
            过滤后的点云 (M x 3)
        """
        mask = confidences >= min_confidence
        return points[mask]

    def statistical_outlier_removal(self, 
                                  points: np.ndarray,
                                  nb_neighbors: int = 20,
                                  std_ratio: float = 2.0) -> np.ndarray:
        """
        统计离群点去除

        Args:
            points: 输入点云 (N x 3)
            nb_neighbors: 邻居数量
            std_ratio: 标准差比率

        Returns:
            过滤后的点云 (M x 3)
        """
        if len(points) <= max(3, nb_neighbors):
            return points
        if pcl is not None:
            cloud = pcl.PointCloud()
            cloud.from_array(points.astype(np.float32))
            sor = cloud.make_statistical_outlier_filter()
            sor.set_mean_k(nb_neighbors)
            sor.set_std_dev_mul_thresh(std_ratio)
            filtered_cloud = sor.filter()
            return np.array(filtered_cloud)

        k = int(max(3, min(nb_neighbors, len(points) - 1)))
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)
        mean_neighbor_dist = np.mean(dists[:, 1:], axis=1)
        mu = float(np.mean(mean_neighbor_dist))
        sigma = float(np.std(mean_neighbor_dist))
        if sigma <= 1e-12:
            return points
        mask = mean_neighbor_dist <= (mu + float(std_ratio) * sigma)
        kept = points[mask]
        return kept if len(kept) > 0 else points
