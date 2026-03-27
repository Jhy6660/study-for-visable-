"""
法向量估计模块
"""
import numpy as np
from typing import Optional


def orient_normal_inward(surface_point_cam: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    将平面法向量定向为指向物体内部（与 pose_estimation 中「n 指向立方体内部」一致）。

    深度/SVD 得到的法向量为二义性 ±n。约定：从侧面上某点指向相机方向为「朝外」，
    则「朝内」与朝外相反。实现：令 to_cam 为从表面指向相机的单位向量，
    若 dot(n, to_cam) > 0 则 n 朝外，取 n = -n。

    Args:
        surface_point_cam: 表面点在相机坐标系下的位置 [X,Y,Z]
        normal: 未定向的单位法向量（任意符号）

    Returns:
        指向物体内部的单位法向量（若输入退化则原样返回）
    """
    p = np.asarray(surface_point_cam, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    pn = np.linalg.norm(p)
    nn = np.linalg.norm(n)
    if pn < 1e-9 or nn < 1e-9:
        return normal
    to_cam = -p / pn
    n = n / nn
    if np.dot(n, to_cam) > 0.0:
        n = -n
    return n


def estimate_normal_from_depth(depth_image: np.ndarray, u: float, v: float,
                               fx: float, fy: float, cx: float, cy: float,
                               window: int = 7) -> Optional[np.ndarray]:
    """
    从深度图估计法向量（多点平均提升稳定性）
    
    Args:
        depth_image: 深度图
        u, v: 像素坐标
        fx, fy, cx, cy: 相机内参
        window: 窗口大小
        
    Returns:
        单位法向量或None
    """
    if depth_image is None:
        return None
    
    h, w = depth_image.shape
    u0, v0 = int(round(u)), int(round(v))
    r = window // 2
    u_min = max(0, u0 - r)
    u_max = min(w, u0 + r + 1)
    v_min = max(0, v0 - r)
    v_max = min(h, v0 + r + 1)
    
    # 3x3网格多点法向量
    grid_size = 3
    normals = []
    
    roi_width = u_max - u_min
    roi_height = v_max - v_min
    
    for i in range(grid_size):
        for j in range(grid_size):
            grid_u = u_min + (i + 0.5) * roi_width / grid_size
            grid_v = v_min + (j + 0.5) * roi_height / grid_size
            
            local_window = max(3, window // 2)
            local_normal = _compute_local_normal(depth_image, grid_u, grid_v, 
                                                 local_window, fx, fy, cx, cy)
            if local_normal is not None:
                normals.append(local_normal)
    
    if len(normals) < 3:
        return _compute_local_normal(depth_image, u, v, window, fx, fy, cx, cy)
    
    # 法向量平均（确保同向）
    normals = np.array(normals)
    reference_normal = normals[0]
    for i in range(1, len(normals)):
        if np.dot(normals[i], reference_normal) < 0:
            normals[i] = -normals[i]
    
    mean_normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(mean_normal)
    
    if norm > 1e-6:
        return mean_normal / norm
    
    return None


def _compute_local_normal(depth_image: np.ndarray, u: float, v: float, window: int,
                          fx: float, fy: float, cx: float, cy: float) -> Optional[np.ndarray]:
    """计算局部法向量"""
    h, w = depth_image.shape
    u0, v0 = int(round(u)), int(round(v))
    r = window // 2
    u_min = max(0, u0 - r)
    u_max = min(w, u0 + r + 1)
    v_min = max(0, v0 - r)
    v_max = min(h, v0 + r + 1)
    
    patch = depth_image[v_min:v_max, u_min:u_max]
    valid_mask = (patch > 0.1) & (patch < 10.0) & np.isfinite(patch)
    ys, xs = np.where(valid_mask)
    
    if len(xs) < 6:
        return None
    
    zs = patch[ys, xs]
    us = xs + u_min
    vs = ys + v_min
    
    # 转换为3D点
    xs_3d = (us - cx) * zs / fx
    ys_3d = (vs - cy) * zs / fy
    points = np.column_stack([xs_3d, ys_3d, zs])
    
    # SVD计算法向量
    try:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal = orient_normal_inward(centroid, normal)
        return normal
    except np.linalg.LinAlgError:
        return None


def estimate_normal_simple(
    depth_image: np.ndarray,
    u: float,
    v: float,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
) -> np.ndarray:
    """
    简化法向量估计（使用深度梯度）
    
    Args:
        depth_image: 深度图
        u, v: 像素坐标
        fx, fy, cx, cy: 若提供，则在相机坐标系下对法向量做「朝内」定向
        
    Returns:
        单位法向量（约定指向物体内部，若无法定向则与深度梯度一致）
    """
    h, w = depth_image.shape
    u0, v0 = int(round(u)), int(round(v))
    
    # 边界检查
    if u0 < 1 or u0 >= w - 1 or v0 < 1 or v0 >= h - 1:
        return np.array([0.0, 0.0, -1.0])
    
    try:
        # 计算深度梯度
        dzdx = depth_image[v0, u0 + 1] - depth_image[v0, u0 - 1]
        dzdy = depth_image[v0 + 1, u0] - depth_image[v0 - 1, u0]
        
        if np.isfinite(dzdx) and np.isfinite(dzdy):
            normal = np.array([-dzdx, -dzdy, 1.0])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                n = normal / norm
                if (
                    fx is not None
                    and fy is not None
                    and cx is not None
                    and cy is not None
                ):
                    z0 = float(depth_image[v0, u0])
                    if z0 > 1e-6:
                        X = (u0 - cx) * z0 / fx
                        Y = (v0 - cy) * z0 / fy
                        p = np.array([X, Y, z0])
                        n = orient_normal_inward(p, n)
                return n
    except Exception:
        pass
    
    # 默认法向量
    return np.array([0.0, 0.0, -1.0])


def validate_normal_with_gravity(
    normal: np.ndarray,
    gravity_vector: np.ndarray,
    angle_threshold: float = 45.0,
    expect_side_face: bool = True,
) -> tuple:
    """
    使用重力向量验证法向量

    Args:
        normal: 法向量
        gravity_vector: 重力向量（相机坐标系）
        angle_threshold: 最大允许偏差（度）
        expect_side_face: True 时按「立方体侧面」处理：法向量应与重力近似垂直（夹角约 90°）；
            False 时按「法向量与重力方向接近」处理（夹角不超过 angle_threshold，适合顶面等）。

    Returns:
        (is_valid, angle_degrees) — angle_degrees 为法向量与重力向量的夹角 [0, 180]
    """
    normal_norm = normal / np.linalg.norm(normal)
    gravity_norm = gravity_vector / np.linalg.norm(gravity_vector)

    dot_product = np.dot(normal_norm, gravity_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(dot_product))

    if expect_side_face:
        # 侧面：n ⊥ g，夹角应接近 90°
        is_valid = abs(angle_degrees - 90.0) <= angle_threshold
    else:
        is_valid = angle_degrees <= angle_threshold

    return is_valid, angle_degrees


class NormalEstimatorWithSmoothing:
    """带平滑的法向量估计器"""
    
    def __init__(
        self,
        window_size: int = 5,
        expect_side_face: bool = True,
        max_interframe_angle_deg: float = 0.0,
    ):
        """
        初始化法向量估计器
        
        Args:
            window_size: 平滑窗口大小
            expect_side_face: 与 validate_normal_with_gravity 一致，用于置信度与验证
            max_interframe_angle_deg: >0 时若与上一帧法向夹角超过该值则清空历史（抑制跳变）
        """
        self.window_size = window_size
        self.expect_side_face = expect_side_face
        self.max_interframe_angle_deg = float(max_interframe_angle_deg)
        self.normal_history = []
        self.confidence_history = []
    
    def update(self, normal: np.ndarray, 
               gravity_vector: np.ndarray = None) -> tuple:
        """
        更新法向量估计
        
        Args:
            normal: 新的法向量
            gravity_vector: 重力向量（可选）
            
        Returns:
            (smoothed_normal, confidence)
        """
        # 归一化法向量
        if np.linalg.norm(normal) > 1e-6:
            normal = normal / np.linalg.norm(normal)

        # 帧间大跳变：清空历史，避免错误平滑
        if self.max_interframe_angle_deg > 1e-6 and len(self.normal_history) > 0:
            prev = self.normal_history[-1]
            dotp = np.clip(np.abs(np.dot(normal, prev)), 0.0, 1.0)
            ang = np.degrees(np.arccos(dotp))
            if ang > self.max_interframe_angle_deg:
                self.reset()
        
        # 计算置信度
        confidence = 1.0
        if gravity_vector is not None:
            _, angle = validate_normal_with_gravity(
                normal, gravity_vector, expect_side_face=self.expect_side_face
            )
            if self.expect_side_face:
                # 越接近 90°（与重力垂直）置信度越高
                confidence = max(0.1, 1.0 - abs(angle - 90.0) / 90.0)
            else:
                confidence = max(0.1, 1.0 - angle / 90.0)
        
        # 添加到历史记录
        self.normal_history.append(normal)
        self.confidence_history.append(confidence)
        
        # 限制历史记录长度
        if len(self.normal_history) > self.window_size:
            self.normal_history.pop(0)
            self.confidence_history.pop(0)
        
        # 加权平均平滑
        if len(self.normal_history) > 0:
            weights = np.array(self.confidence_history)
            weights = weights / np.sum(weights)
            
            # 确保法向量方向一致
            reference = self.normal_history[0]
            for i in range(1, len(self.normal_history)):
                if np.dot(self.normal_history[i], reference) < 0:
                    self.normal_history[i] = -self.normal_history[i]
            
            # 加权平均
            smoothed = np.average(
                np.array(self.normal_history), 
                axis=0, 
                weights=weights
            )
            
            # 归一化
            if np.linalg.norm(smoothed) > 1e-6:
                smoothed = smoothed / np.linalg.norm(smoothed)
            
            return smoothed, confidence
        
        return normal, confidence
    
    def reset(self):
        """重置估计器"""
        self.normal_history.clear()
        self.confidence_history.clear()
