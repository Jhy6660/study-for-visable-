"""
立方体定位模块 - 相机+雷达融合定位

方案3实现：
1. 相机检测图案 → 得到图案在图像中的位置
2. 雷达点云分割 → 得到立方体点云
3. 融合：将图案投影到点云 → 精确定位侧面中心
"""
import numpy as np
from typing import Optional, Tuple, List
try:
    import pcl
except Exception:
    pcl = None
from scipy.spatial import cKDTree

from .normal_estimation import orient_normal_inward


class CubeLocalization:
    """立方体融合定位器"""

    def __init__(self, cube_side_length: float = 0.35):
        """
        初始化立方体定位器

        Args:
            cube_side_length: 立方体边长（米）
        """
        self.cube_side_length = cube_side_length
        self.half_side = cube_side_length / 2  # 175mm

    def localize_from_frustum_and_pointcloud(
        self,
        bbox: List[int],
        camera_params: dict,
        lidar_points: np.ndarray,
        lidar_to_camera_transform: np.ndarray = None,
        bbox_shrink_ratio: float = 0.1,
        min_frustum_points: int = 10,
        min_cube_points: int = 10,
        foreground_depth_range: float = 0.3,
        foreground_depth_percentile: float = 35.0,
    ) -> Optional[dict]:
        """
        [高级算法] Frustum PointNets 思路：
        基于 YOLO 2D 检测框构建视锥体 (Frustum)，直接从雷达点云中提取目标。
        完全摆脱对深度相机深度图的依赖，极大提高远距离和反光表面的定位精度。

        Args:
            bbox: [x_min, y_min, x_max, y_max] YOLO 检测框
            camera_params: 相机内参
            lidar_points: 雷达点云 (N x 3)
            lidar_to_camera_transform: 雷达到相机的外参矩阵

        Returns:
            定位结果字典
        """
        # 1. 转换雷达点云到相机坐标系
        if lidar_to_camera_transform is not None:
            lidar_points_cam = self._transform_pointcloud(lidar_points, lidar_to_camera_transform)
        else:
            lidar_points_cam = lidar_points
            
        # 2. 剔除相机背后的点 (Z <= 0)
        valid_z_mask = lidar_points_cam[:, 2] > 0.1
        pts_cam = lidar_points_cam[valid_z_mask]
        if len(pts_cam) == 0:
            return None
            
        # 3. 将 3D 点投影到 2D 图像平面
        fx, fy = camera_params['fx'], camera_params['fy']
        cx, cy = camera_params['cx'], camera_params['cy']
        
        u = (fx * pts_cam[:, 0] / pts_cam[:, 2]) + cx
        v = (fy * pts_cam[:, 1] / pts_cam[:, 2]) + cy
        
        # 4. 根据 YOLO 2D 框进行视锥体截取 (Frustum Cropping)
        x_min, y_min, x_max, y_max = bbox
        bbox_shrink_ratio = float(np.clip(bbox_shrink_ratio, 0.0, 0.45))
        margin_x = (x_max - x_min) * bbox_shrink_ratio
        margin_y = (y_max - y_min) * bbox_shrink_ratio
        
        in_box_mask = (u >= (x_min + margin_x)) & (u <= (x_max - margin_x)) & \
                      (v >= (y_min + margin_y)) & (v <= (y_max - margin_y))
                      
        frustum_points = pts_cam[in_box_mask]
        
        if len(frustum_points) < int(min_frustum_points):
            return None
            
        # 5. 去除背景点：同时使用“最近深度窗口”与“分位数深度”两种门限，增强稳定性
        z_values = frustum_points[:, 2]
        min_z = np.min(z_values)
        depth_gate_range = min_z + max(float(foreground_depth_range), 0.05)
        p = float(np.clip(foreground_depth_percentile, 5.0, 90.0))
        depth_gate_percentile = np.percentile(z_values, p)
        depth_gate = min(depth_gate_range, depth_gate_percentile)
        foreground_mask = z_values <= depth_gate
        cube_points = frustum_points[foreground_mask]
        
        if len(cube_points) < int(min_cube_points):
            return None
            
        # 统计滤波去除离群点
        cube_points = self._statistical_outlier_removal(cube_points)
        if len(cube_points) < 6:
            return None

        # 6. 平面拟合
        plane_result = self._fit_plane_from_points(cube_points)
        if plane_result is None:
            return None
            
        side_center, normal, plane_inliers = plane_result
        
        # 图案中心 2D 坐标
        pattern_u = (x_min + x_max) / 2.0
        pattern_v = (y_min + y_max) / 2.0
        
        # 将图案中心沿射线投影到拟合出的平面上
        # 射线方向向量: ray = [ (u-cx)/fx, (v-cy)/fy, 1.0 ]
        ray = np.array([(pattern_u - cx) / fx, (pattern_v - cy) / fy, 1.0])
        ray = ray / np.linalg.norm(ray)
        
        # 射线与平面交点: t = dot(normal, side_center) / dot(normal, ray)
        dot_nr = np.dot(normal, ray)
        if abs(dot_nr) < 1e-6:
            corrected_side_center = side_center
        else:
            t = np.dot(normal, side_center) / dot_nr
            corrected_side_center = side_center if t <= 0.0 else (ray * t)
            
        # 7. 计算立方体中心
        cube_center = corrected_side_center + normal * self.half_side
        
        inlier_ratio = len(plane_inliers) / max(len(cube_points), 1)
        depth_span = float(np.max(z_values) - np.min(z_values))
        depth_consistency = float(np.exp(-depth_span / 0.6))
        confidence = float(np.clip(0.8 * inlier_ratio + 0.2 * depth_consistency, 0.0, 1.0))
        
        return {
            'side_center': corrected_side_center,
            'cube_center': cube_center,
            'normal': normal,
            'confidence': confidence,
            'num_points': len(cube_points)
        }

    def localize_from_pattern_and_pointcloud(
        self,
        pattern_uv: Tuple[float, float],
        pattern_depth: float,
        camera_params: dict,
        lidar_points: np.ndarray,
        lidar_to_camera_transform: np.ndarray = None
    ) -> Optional[dict]:
        """
        融合定位：相机图案 + 雷达点云

        Args:
            pattern_uv: 图案在图像中的像素坐标 (u, v)
            pattern_depth: 图案处的深度值（米）
            camera_params: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            lidar_points: 雷达点云 (N x 3)，在雷达坐标系下
            lidar_to_camera_transform: 雷达到相机的变换矩阵 (4x4)

        Returns:
            定位结果字典，包含：
            - side_center: 侧面中心（相机坐标系）
            - cube_center: 立方体中心（相机坐标系）
            - normal: 侧面法向量
            - confidence: 置信度
        """
        # 1. 将图案像素转换为相机坐标系下的3D点
        pattern_3d_cam = self._pixel_to_camera_3d(
            pattern_uv[0], pattern_uv[1], pattern_depth, camera_params
        )

        # 2. 将雷达点云转换到相机坐标系
        if lidar_to_camera_transform is not None:
            lidar_points_cam = self._transform_pointcloud(
                lidar_points, lidar_to_camera_transform
            )
        else:
            lidar_points_cam = lidar_points

        # 3. 在点云中找到立方体附近的点
        cube_points = self._extract_cube_points(
            lidar_points_cam, pattern_3d_cam
        )

        if len(cube_points) < 20:
            return None

        # 4. 从点云拟合侧面平面
        plane_result = self._fit_plane_from_points(cube_points)

        if plane_result is None:
            return None

        side_center, normal, plane_inliers = plane_result

        # 5. 校正侧面中心：使用图案位置校正
        corrected_side_center = self._correct_side_center(
            side_center, pattern_3d_cam, normal
        )

        # 6. 计算立方体中心
        cube_center = corrected_side_center + normal * self.half_side

        # 7. 计算置信度
        confidence = self._compute_confidence(
            cube_points, plane_inliers, pattern_3d_cam, corrected_side_center
        )

        return {
            'side_center': corrected_side_center,
            'cube_center': cube_center,
            'normal': normal,
            'confidence': confidence,
            'num_points': len(cube_points)
        }

    def _pixel_to_camera_3d(self, u: float, v: float, depth: float,
                           camera_params: dict) -> np.ndarray:
        """像素坐标转相机3D坐标"""
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def _transform_pointcloud(self, points: np.ndarray,
                             transform: np.ndarray) -> np.ndarray:
        """变换点云坐标系"""
        ones = np.ones((points.shape[0], 1))
        points_hom = np.hstack([points, ones])
        transformed = (transform @ points_hom.T).T
        return transformed[:, :3]

    def _extract_cube_points(self, lidar_points: np.ndarray,
                            pattern_3d: np.ndarray,
                            search_radius: float = 0.3) -> np.ndarray:
        """
        从雷达点云中提取立方体附近的点

        Args:
            lidar_points: 雷达点云（相机坐标系）
            pattern_3d: 图案3D位置
            search_radius: 搜索半径（米）

        Returns:
            立方体附近的点云
        """
        # 计算每个点到图案的距离
        distances = np.linalg.norm(lidar_points - pattern_3d, axis=1)

        # 选择在搜索半径内的点
        mask = distances < search_radius
        cube_points = lidar_points[mask]

        # 进一步过滤：只保留在立方体尺寸范围内的点
        if len(cube_points) > 0:
            # 使用统计离群点去除
            cube_points = self._statistical_outlier_removal(cube_points)

        return cube_points

    def _statistical_outlier_removal(self, points: np.ndarray,
                                     nb_neighbors: int = 10,
                                     std_ratio: float = 1.5) -> np.ndarray:
        """统计离群点去除"""
        if len(points) < nb_neighbors:
            return points

        try:
            if pcl is not None:
                cloud = pcl.PointCloud()
                cloud.from_array(points.astype(np.float32))

                sor = cloud.make_statistical_outlier_filter()
                sor.set_mean_k(nb_neighbors)
                sor.set_std_dev_mul_thresh(std_ratio)

                filtered = sor.filter()
                return np.array(filtered)

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
        except Exception:
            return points

    def _fit_plane_from_points(self, points: np.ndarray) -> Optional[Tuple]:
        """
        从点云拟合平面 (使用 PyTorch 加速的 RANSAC)

        Returns:
            (side_center, normal, inliers) 或 None
        """
        if len(points) < 6:
            return None

        try:
            import torch
            if torch.cuda.is_available():
                return self._fit_plane_ransac_pytorch(points)
            else:
                return self._fit_plane_svd(points)
        except Exception as e:
            print(f"RANSAC failed, falling back to SVD: {e}")
            return self._fit_plane_svd(points)

    def _fit_plane_ransac_pytorch(self, points: np.ndarray, distance_threshold=0.015, max_iterations=200) -> Optional[Tuple]:
        import torch
        device = torch.device('cuda')
        pts = torch.tensor(points, dtype=torch.float32, device=device)
        n_points = pts.shape[0]
        
        if n_points < 10:
            return self._fit_plane_svd(points)
            
        # 随机采样
        indices = torch.randint(0, n_points, (max_iterations, 3), device=device)
        p0 = pts[indices[:, 0]]
        p1 = pts[indices[:, 1]]
        p2 = pts[indices[:, 2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normals = torch.cross(v1, v2, dim=1)
        norms = torch.norm(normals, dim=1, keepdim=True)
        
        valid_mask = norms.squeeze() > 1e-6
        if not valid_mask.any():
            return self._fit_plane_svd(points)
            
        normals = normals[valid_mask] / norms[valid_mask]
        p0 = p0[valid_mask]
        
        # 计算所有点到这些平面的距离
        pts_dot_n = torch.matmul(pts, normals.t())
        p0_dot_n = torch.sum(p0 * normals, dim=1)
        distances = torch.abs(pts_dot_n - p0_dot_n.unsqueeze(0))
        
        inliers = distances < distance_threshold
        inlier_counts = torch.sum(inliers, dim=0)
        
        best_idx = torch.argmax(inlier_counts)
        best_inlier_mask = inliers[:, best_idx]
        
        best_pts = pts[best_inlier_mask].cpu().numpy()
        
        if len(best_pts) < 10:
            return self._fit_plane_svd(points)
            
        # 对最佳 inliers 再做一次 SVD 获得更精准的法向量
        return self._fit_plane_svd(best_pts)

    def _fit_plane_svd(self, points: np.ndarray) -> Optional[Tuple]:
        """
        使用传统的 SVD 拟合平面
        """
        try:
            # 使用SVD拟合平面
            centroid = np.mean(points, axis=0)
            centered = points - centroid

            # SVD分解
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # 最小奇异值对应的向量是法向量（符号待定，下面用侧面中心定向）
            normal = Vt[-1]

            # 计算内点（距离平面小于阈值的点；|dot| 与法向量符号无关）
            distances = np.abs(np.dot(centered, normal))
            inlier_threshold = 0.015  # 1.5cm
            inliers = points[distances < inlier_threshold]

            if len(inliers) < 10:
                return None

            # 重新用内点计算侧面中心，再定向为指向立方体内部（与 pose_estimation 一致）
            side_center = np.mean(inliers, axis=0)
            normal = orient_normal_inward(side_center, normal)

            return side_center, normal, inliers

        except np.linalg.LinAlgError:
            return None

    def _correct_side_center(self, side_center: np.ndarray,
                            pattern_3d: np.ndarray,
                            normal: np.ndarray) -> np.ndarray:
        """
        校正侧面中心位置

        使用图案位置校正点云拟合得到的侧面中心，
        消除图案不在侧面正中心带来的偏差。

        Args:
            side_center: 点云拟合的侧面中心
            pattern_3d: 图案的3D位置
            normal: 侧面法向量

        Returns:
            校正后的侧面中心
        """
        # 将图案位置投影到侧面上
        # 沿法向量方向调整，使图案在平面上

        # 计算图案到平面的距离
        dist_to_plane = np.dot(pattern_3d - side_center, normal)

        # 投影图案到平面
        pattern_on_plane = pattern_3d - dist_to_plane * normal

        # 假设图案在侧面中心附近，使用加权平均
        # 权重：点云拟合结果 0.3，图案位置 0.7
        # 因为相机检测图案精度更高
        corrected = 0.3 * side_center + 0.7 * pattern_on_plane

        return corrected

    def _compute_confidence(self, cube_points: np.ndarray,
                           inliers: np.ndarray,
                           pattern_3d: np.ndarray,
                           side_center: np.ndarray) -> float:
        """
        计算定位置信度

        Args:
            cube_points: 立方体点云
            inliers: 平面内点
            pattern_3d: 图案位置
            side_center: 侧面中心

        Returns:
            置信度 [0, 1]
        """
        confidence = 1.0

        # 1. 点云数量因子
        num_points = len(cube_points)
        if num_points < 20:
            confidence *= 0.5
        elif num_points < 50:
            confidence *= 0.7
        elif num_points < 100:
            confidence *= 0.9

        # 2. 平面拟合质量因子
        inlier_ratio = len(inliers) / max(num_points, 1)
        confidence *= inlier_ratio

        # 3. 图案与侧面中心一致性因子
        dist_pattern_to_center = np.linalg.norm(pattern_3d - side_center)
        # 图案应该在侧面中心附近（假设偏差不超过立方体边长的30%）
        max_expected_dist = self.cube_side_length * 0.3
        if dist_pattern_to_center > max_expected_dist:
            consistency = max_expected_dist / dist_pattern_to_center
            confidence *= consistency

        return min(max(confidence, 0.0), 1.0)

    def compute_grasp_position(self, localization_result: dict,
                              gravity: np.ndarray = np.array([0.0, 0.0, -1.0])
                              ) -> np.ndarray:
        """
        从定位结果计算抓取位置（顶面中心）

        Args:
            localization_result: localize_from_pattern_and_pointcloud的返回结果
            gravity: 重力向量（相机坐标系）

        Returns:
            抓取位置（顶面中心）
        """
        side_center = localization_result['side_center']
        normal = localization_result['normal']

        # 使用改进的方法计算顶面中心
        from .pose_estimation import compute_rotation_matrix_from_normal

        R = compute_rotation_matrix_from_normal(normal, gravity)
        a = R[:, 0]  # 平面内参考轴
        n = R[:, 2]  # 法向量

        # p_top = p_side + (L/2) * (a - n)
        grasp_position = side_center + self.half_side * (a - n)

        return grasp_position


class MultiCandidateGrasp:
    """多候选抓取位置生成器"""

    def __init__(self, cube_side_length: float = 0.35):
        self.cube_side_length = cube_side_length

    def generate_candidates(self, localization_result: dict,
                           gravity: np.ndarray,
                           num_candidates: int = 4) -> List[dict]:
        """
        生成多个候选抓取位置

        Args:
            localization_result: 定位结果
            gravity: 重力向量
            num_candidates: 候选数量

        Returns:
            候选抓取位置列表，每个包含：
            - position: 抓取位置
            - approach_direction: 接近方向
            - score: 评分
        """
        from .pose_estimation import compute_rotation_matrix_from_normal

        side_center = localization_result['side_center']
        normal = localization_result['normal']
        confidence = localization_result['confidence']

        R = compute_rotation_matrix_from_normal(normal, gravity)
        a = R[:, 0]
        n = R[:, 2]

        candidates = []
        half_side = self.cube_side_length / 2

        # 候选1：顶面中心（推荐）
        top_center = side_center + half_side * (a - n)
        candidates.append({
            'position': top_center,
            'approach_direction': -n,  # 从侧面接近
            'score': confidence * 1.0,
            'type': 'top_center'
        })

        # 候选2-N：顶面中心周围的位置
        if num_candidates > 1:
            angles = np.linspace(0, 2*np.pi, num_candidates, endpoint=False)
            for i, angle in enumerate(angles[1:], start=1):
                # 在顶面上绕中心旋转
                offset = half_side * 0.3 * np.array([
                    np.cos(angle), np.sin(angle), 0
                ])
                # 将偏移转换到立方体坐标系
                offset_world = R @ offset

                candidate_pos = top_center + offset_world
                candidates.append({
                    'position': candidate_pos,
                    'approach_direction': -n,
                    'score': confidence * (1.0 - 0.1 * i),  # 稍低的评分
                    'type': f'top_offset_{i}'
                })

        # 按评分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates
