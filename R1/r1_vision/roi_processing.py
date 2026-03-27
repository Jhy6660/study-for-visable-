"""
ROI处理模块 - 图案中心定位
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def find_pattern_center(image: np.ndarray, bbox: np.ndarray) -> Tuple[float, float]:
    """
    在检测框内定位图案中心
    
    Args:
        image: BGR图像
        bbox: 检测框 [x1, y1, x2, y2]
        
    Returns:
        图案中心坐标 (cx, cy)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    # 提取ROI
    roi = image[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # 方法1: 轮廓检测（适合黑白图案）
    center = _find_center_by_contour(roi_gray, x1, y1)
    if center is not None:
        return center
    
    # 方法2: 角点检测（备选）
    center = _find_center_by_corners(roi_gray, x1, y1)
    if center is not None:
        return center
    
    # 回退到bbox中心
    return (x1 + x2) / 2, (y1 + y2) / 2


def _find_center_by_contour(roi_gray: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[float, float]]:
    """通过轮廓检测找中心"""
    try:
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx_roi = M["m10"] / M["m00"]
                cy_roi = M["m01"] / M["m00"]
                return offset_x + cx_roi, offset_y + cy_roi
    except Exception:
        pass
    return None


def _find_center_by_corners(roi_gray: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[float, float]]:
    """通过角点检测找中心"""
    try:
        corners = cv2.goodFeaturesToTrack(roi_gray, maxCorners=10, qualityLevel=0.01, minDistance=10)
        if corners is not None and len(corners) > 0:
            center_roi = np.mean(corners.reshape(-1, 2), axis=0)
            return offset_x + center_roi[0], offset_y + center_roi[1]
    except Exception:
        pass
    return None


def calculate_gray_variance(image: np.ndarray, bbox: np.ndarray) -> float:
    """
    计算检测框内的灰度方差（用于光照自适应）
    
    Args:
        image: BGR图像
        bbox: 检测框 [x1, y1, x2, y2]
        
    Returns:
        灰度方差值
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    return float(np.var(gray, dtype=np.float32))
