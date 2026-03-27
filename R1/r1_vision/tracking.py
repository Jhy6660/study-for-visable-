"""
目标跟踪模块 - Kalman滤波
"""
import cv2
import time
import numpy as np
from collections import deque
from typing import Optional


class KalmanTracker:
    """改进的Kalman跟踪器 - 动态dt，单目标优化"""
    
    def __init__(self, initial_pos: np.ndarray, track_id: int = 0):
        """
        初始化跟踪器
        
        Args:
            initial_pos: 初始3D位置 [x, y, z]
            track_id: 跟踪ID
        """
        self.track_id = track_id
        self.kf = cv2.KalmanFilter(6, 3)  # 状态: [x,y,z,vx,vy,vz], 观测: [x,y,z]
        
        self.dt = 1.0 / 30.0
        self.last_time = time.time()
        
        # 状态转移矩阵（匀速模型）
        self._update_transition_matrix()
        
        # 观测矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 噪声协方差
        self.kf.processNoiseCov = np.ascontiguousarray(np.eye(6, dtype=np.float32) * 0.01)
        self.kf.measurementNoiseCov = np.ascontiguousarray(np.eye(3, dtype=np.float32) * 0.1)
        
        # 初始状态
        self.kf.statePost = np.array([
            initial_pos[0], initial_pos[1], initial_pos[2], 0, 0, 0
        ], dtype=np.float32).reshape(6, 1)
        self.kf.errorCovPost = np.ascontiguousarray(np.eye(6, dtype=np.float32) * 0.1)
        
        self.last_update = time.time()
        self.missed_frames = 0
        self.confidence_history = deque(maxlen=10)
    
    def _update_transition_matrix(self):
        """动态更新状态转移矩阵"""
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
    
    def predict(self) -> np.ndarray:
        """预测下一帧位置"""
        current_time = time.time()
        self.dt = current_time - self.last_time
        self.dt = max(0.01, min(0.2, self.dt))  # 限制dt范围
        
        self._update_transition_matrix()
        prediction = self.kf.predict()
        return prediction.flatten()[:3]
    
    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """更新跟踪器"""
        current_time = time.time()
        measurement_reshaped = measurement.reshape(3, 1).astype(np.float32)
        self.kf.correct(measurement_reshaped)
        
        self.last_update = current_time
        self.last_time = current_time
        self.missed_frames = 0
        self.confidence_history.append(confidence)
    
    def get_confidence(self) -> float:
        """获取跟踪置信度"""
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))
    
    def is_valid(self, max_missed_frames: int = 10, max_age: float = 2.0) -> bool:
        """检查跟踪器是否有效"""
        age = time.time() - self.last_update
        return self.missed_frames < max_missed_frames and age < max_age
