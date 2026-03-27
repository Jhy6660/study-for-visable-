"""
R1 Vision - 模块化视觉检测系统

基于YOLO和深度相机的目标检测、跟踪与位姿估计系统
支持GPU加速、Kalman滤波、光照自适应等高级功能
"""

__version__ = "2.0.0"
__author__ = "R1 Developer"
__email__ = "developer@example.com"

from .detection import ObjectDetector
from .depth_processing import get_median_depth, depth_roi_filtering
from .tracking import KalmanTracker
from .comms import SerialComm
from .roi_processing import find_pattern_center, calculate_gray_variance
from .normal_estimation import estimate_normal_from_depth, estimate_normal_simple
from .pose_estimation import (
    pixel_to_camera, 
    compute_grasp_position, 
    compute_grasp_position_improved,
    compute_rotation_matrix_from_normal,
    compute_top_center_from_side,
    validate_position
)
from .utils import PerformanceMonitor, format_position, setup_device

__all__ = [
    'ObjectDetector',
    'get_median_depth',
    'depth_roi_filtering', 
    'KalmanTracker',
    'SerialComm',
    'find_pattern_center',
    'calculate_gray_variance',
    'estimate_normal_from_depth',
    'estimate_normal_simple',
    'pixel_to_camera',
    'compute_grasp_position',
    'validate_position',
    'PerformanceMonitor',
    'format_position',
    'setup_device',
]
