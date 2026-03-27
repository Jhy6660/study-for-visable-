"""
位姿估计模块 - 像素到3D坐标转换和抓取位置计算
"""
import numpy as np
from typing import Tuple, Optional, Dict


def pixel_to_camera(u: float, v: float, depth: float, 
                   fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    像素坐标转相机坐标
    
    Args:
        u, v: 像素坐标
        depth: 深度值（米）
        fx, fy, cx, cy: 相机内参
        
    Returns:
        相机坐标系3D点 [X, Y, Z]
    """
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def compute_grasp_position(face_point: np.ndarray, normal: np.ndarray, 
                          normal_offset: float = 0.05, z_offset: float = 0.125) -> np.ndarray:
    """
    计算抓取位置
    
    Args:
        face_point: 物体表面点
        normal: 表面法向量
        normal_offset: 沿法向量偏移距离（米）
        z_offset: Z轴额外偏移（米）
        
    Returns:
        抓取位置 [X, Y, Z]
    """
    grasp_pos = face_point + normal_offset * normal
    grasp_pos[2] += z_offset  # Z轴额外偏移
    return grasp_pos


def compute_cube_center(face_point: np.ndarray, normal: np.ndarray, 
                       half_size: float = 0.175) -> np.ndarray:
    """
    计算立方体中心位置
    
    Args:
        face_point: 表面点
        normal: 法向量
        half_size: 立方体半边长（米）
        
    Returns:
        立方体中心 [X, Y, Z]
    """
    return face_point + normal * half_size


def validate_position(position: np.ndarray, 
                     min_bounds: np.ndarray = np.array([-2.0, -2.0, 0.1]),
                     max_bounds: np.ndarray = np.array([2.0, 2.0, 3.0])) -> bool:
    """
    验证3D位置是否在合理范围内
    
    Args:
        position: 3D位置
        min_bounds: 最小边界
        max_bounds: 最大边界
        
    Returns:
        是否有效
    """
    return np.all(position >= min_bounds) and np.all(position <= max_bounds)


def compute_rotation_matrix_from_normal(normal: np.ndarray, 
                                       gravity: np.ndarray = np.array([0.0, 0.0, -1.0])) -> np.ndarray:
    """
    从法向量和重力向量构造旋转矩阵

    根据报告中的方法：
    1. 将重力向量投影到平面内得到参考轴 a
    2. 计算 b = n × a
    3. 构造旋转矩阵 R = [a, b, n]

    Args:
        normal: 单位法向量（指向立方体内部）
        gravity: 重力向量（相机坐标系下），默认向下(0,0,-1)

    Returns:
        3x3 旋转矩阵 R = [a, b, n]
    """
    # 确保法向量是单位向量
    n = normal / np.linalg.norm(normal)
    g = gravity / np.linalg.norm(gravity)

    # 计算平面内参考轴 a：将重力投影到平面内
    a_prime = g - np.dot(g, n) * n
    a_norm = np.linalg.norm(a_prime)

    # 如果重力与法向量几乎平行，使用备用方案
    if a_norm < 1e-6:
        # 使用 (0,0,1) × n 作为参考方向
        a_prime = np.array([0.0, 0.0, 1.0])
        a_prime = a_prime - np.dot(a_prime, n) * n
        a_norm = np.linalg.norm(a_prime)

        # 如果仍然太小，使用 (1,0,0) × n
        if a_norm < 1e-6:
            a_prime = np.array([1.0, 0.0, 0.0])
            a_prime = a_prime - np.dot(a_prime, n) * n
            a_norm = np.linalg.norm(a_prime)

    a = a_prime / a_norm

    # 计算第三轴 b = n × a
    b = np.cross(n, a)
    b = b / np.linalg.norm(b)

    # 构造旋转矩阵 R = [a, b, n]
    R = np.column_stack([a, b, n])

    # 确保是右手坐标系
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]  # 翻转 a 的符号

    return R


def compute_top_center_from_side(p_side: np.ndarray, 
                                normal: np.ndarray,
                                cube_side_length: float = 0.35,
                                gravity: np.ndarray = np.array([0.0, 0.0, -1.0])) -> np.ndarray:
    """
    从侧面中心计算顶面中心

    根据报告中的公式：
    p_top = p_side + (L/2) * (a - n)

    其中：
    - p_side: 侧面中心
    - L: 立方体边长
    - n: 侧面法向量（指向立方体内部）
    - a: 平面内参考轴（由重力向量投影得到）

    Args:
        p_side: 侧面中心3D坐标
        normal: 侧面法向量（指向立方体内部）
        cube_side_length: 立方体边长（米），默认 350mm
        gravity: 重力向量（相机坐标系下），默认向下(0,0,-1)

    Returns:
        顶面中心3D坐标
    """
    # 构造旋转矩阵
    R = compute_rotation_matrix_from_normal(normal, gravity)

    # 提取 a 和 n
    a = R[:, 0]
    n = R[:, 2]

    # 计算顶面中心
    p_top = p_side + 0.5 * cube_side_length * (a - n)

    return p_top


def compute_grasp_position_improved(face_point: np.ndarray,
                                   normal: np.ndarray,
                                   cube_side_length: float = 0.35,
                                   gravity: np.ndarray = np.array([0.0, 0.0, -1.0])) -> np.ndarray:
    """
    改进的抓取位置计算（基于平面拟合和重力向量）

    使用报告中的方法：
    1. 从侧面点云拟合平面得到法向量 n
    2. 利用重力向量 g 构造旋转矩阵 R = [a, b, n]
    3. 计算顶面中心：p_top = p_side + (L/2)*(a - n)

    Args:
        face_point: 物体表面点（侧面中心）
        normal: 表面法向量（指向立方体内部）
        cube_side_length: 立方体边长（米），默认 350mm
        gravity: 重力向量（相机坐标系下），默认向下(0,0,-1)

    Returns:
        抓取位置（顶面中心）[X, Y, Z]
    """
    return compute_top_center_from_side(face_point, normal, cube_side_length, gravity)


def validate_normal_direction(normal: np.ndarray, 
                             expected_direction: np.ndarray) -> Tuple[bool, float]:
    """
    验证法向量是否指向预期方向

    Args:
        normal: 法向量
        expected_direction: 预期方向（单位向量）

    Returns:
        (is_valid, dot_product) - 是否合理，点积值
    """
    normal_norm = normal / np.linalg.norm(normal)
    expected_norm = expected_direction / np.linalg.norm(expected_direction)
    dot = np.dot(normal_norm, expected_norm)
    return dot > 0, dot


def estimate_position_error(p_side: np.ndarray, 
                           normal: np.ndarray,
                           gravity: np.ndarray,
                           cube_side_length: float,
                           normal_std: float = 0.05,
                           gravity_std: float = 0.02) -> Dict:
    """
    估计顶面中心位置的误差

    Args:
        p_side: 侧面中心
        normal: 法向量
        gravity: 重力向量
        cube_side_length: 立方体边长
        normal_std: 法向量标准差（弧度）
        gravity_std: 重力向量标准差（弧度）

    Returns:
        误差估计字典
    """
    # 构造旋转矩阵
    R = compute_rotation_matrix_from_normal(normal, gravity)
    a = R[:, 0]
    n = R[:, 2]

    # 误差传播分析
    # p_top = p_side + (L/2)*(a - n)
    # 误差主要来自法向量和重力向量的估计误差

    # 法向误差导致的位置误差
    error_from_normal = (cube_side_length / 2) * normal_std

    # 重力误差导致的位置误差（简化估计）
    error_from_gravity = (cube_side_length / 2) * gravity_std

    # 总误差（假设独立）
    total_error = np.sqrt(error_from_normal**2 + error_from_gravity**2)

    return {
        'error_from_normal': error_from_normal,
        'error_from_gravity': error_from_gravity,
        'total_error': total_error,
        'normal_std_rad': normal_std,
        'gravity_std_rad': gravity_std
    }