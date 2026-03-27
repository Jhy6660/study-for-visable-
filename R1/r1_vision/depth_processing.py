"""
深度处理模块 - 深度滤波和ROI处理
"""
import numpy as np
from typing import Optional, Tuple


def get_median_depth(depth_image: np.ndarray, u: float, v: float, window: int = 5) -> Optional[float]:
    """
    获取窗口中值深度（抗噪声）
    
    Args:
        depth_image: 深度图（米为单位）
        u, v: 像素坐标
        window: 窗口大小
        
    Returns:
        中值深度或None
    """
    if depth_image is None:
        return None
    
    h, w = depth_image.shape
    u0, v0 = int(round(u)), int(round(v))
    
    half_window = window // 2
    u_min = max(0, u0 - half_window)
    u_max = min(w, u0 + half_window + 1)
    v_min = max(0, v0 - half_window)
    v_max = min(h, v0 + half_window + 1)
    
    patch = depth_image[v_min:v_max, u_min:u_max]
    
    # 严格的有效值筛选
    valid_mask = (patch > 0.1) & (patch < 10.0) & np.isfinite(patch)
    valid_depths = patch[valid_mask]
    
    if valid_depths.size < 3:
        return None
    
    # 中值滤波
    median_depth = np.median(valid_depths)
    
    # MAD离群值过滤
    mad = np.median(np.abs(valid_depths - median_depth))
    if mad > 0:
        outlier_mask = np.abs(valid_depths - median_depth) <= 3 * mad
        filtered_depths = valid_depths[outlier_mask]
        if filtered_depths.size >= 3:
            return float(np.median(filtered_depths))
    
    return float(median_depth)


def depth_roi_filtering(depth_image: np.ndarray, bbox: np.ndarray, 
                       center_x: float, center_y: float,
                       fx: float, fy: float, cx: float, cy: float) -> Optional[np.ndarray]:
    """
    ROI深度滤波 - 使用中值或平面拟合
    
    Args:
        depth_image: 深度图
        bbox: 检测框 [x1, y1, x2, y2]
        center_x, center_y: 图案中心
        fx, fy, cx, cy: 相机内参
        
    Returns:
        3D位置 [X, Y, Z] 或 None
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = depth_image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # 提取ROI深度
    roi_depth = depth_image[y1:y2, x1:x2]
    valid_mask = (roi_depth > 0.1) & (roi_depth < 10.0) & np.isfinite(roi_depth)
    
    if np.sum(valid_mask) < 10:
        return None
    
    # 方法1: 中值深度（快速稳定）
    valid_depths = roi_depth[valid_mask]
    
    # 【新增去噪平滑】过滤掉距离相机过近或过远的飞点，取中间核心部分
    p20 = np.percentile(valid_depths, 20)
    p80 = np.percentile(valid_depths, 80)
    core_mask = (valid_depths >= p20) & (valid_depths <= p80)
    if np.sum(core_mask) > 0:
        median_depth = np.median(valid_depths[core_mask])
    else:
        median_depth = np.median(valid_depths)
    
    # 方法2: 平面拟合（更精确，适合平面物体）
    final_depth = median_depth
    try:
        ys, xs = np.where(valid_mask)
        if len(xs) >= 20:  # 足够点数才用平面拟合
            zs = roi_depth[ys, xs]
            us = xs + x1
            vs = ys + y1
            
            # 转换为3D点
            xs_3d = (us - cx) * zs / fx
            ys_3d = (vs - cy) * zs / fy
            points_3d = np.column_stack([xs_3d, ys_3d, zs])
            
            plane_depth = fit_plane_depth(points_3d, center_x, center_y, fx, fy, cx, cy)
            if plane_depth is not None and abs(plane_depth - median_depth) < 0.1:
                final_depth = plane_depth
    except Exception:
        pass
    
    # 计算3D位置
    X = (center_x - cx) * final_depth / fx
    Y = (center_y - cy) * final_depth / fy
    Z = final_depth
    
    return np.array([X, Y, Z])


def fit_plane_depth(points_3d: np.ndarray, query_x: float, query_y: float,
                   fx: float, fy: float, cx: float, cy: float) -> Optional[float]:
    """
    平面拟合计算深度
    
    Args:
        points_3d: 3D点云
        query_x, query_y: 查询点像素坐标
        fx, fy, cx, cy: 相机内参
        
    Returns:
        拟合深度或None
    """
    if len(points_3d) < 10:
        return None
    
    try:
        # 在图像坐标系拟合: depth = a*u + b*v + c
        us = points_3d[:, 0] * fx / points_3d[:, 2] + cx
        vs = points_3d[:, 1] * fy / points_3d[:, 2] + cy
        depths = points_3d[:, 2]
        
        A = np.column_stack([us, vs, np.ones(len(us))])
        params, _, rank, _ = np.linalg.lstsq(A, depths, rcond=None)
        
        if rank < 3:
            return None
        
        a, b, c = params
        predicted_depth = a * query_x + b * query_y + c
        
        if 0.1 < predicted_depth < 10.0:
            return predicted_depth
    except Exception:
        pass
    
    return None
