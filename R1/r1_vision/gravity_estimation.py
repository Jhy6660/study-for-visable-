"""
重力向量估计模块 - 使用卡尔曼滤波和多种平滑方法
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple


class KalmanFilterGravity:
    """用于重力向量的卡尔曼滤波器"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
        """
        # 状态向量：重力向量 [gx, gy, gz]
        self.state = np.array([0.0, 0.0, -1.0])
        
        # 状态协方差矩阵
        self.P = np.eye(3) * 0.1
        
        # 过程噪声协方差
        self.Q = np.eye(3) * process_noise
        
        # 测量噪声协方差
        self.R = np.eye(3) * measurement_noise
        
        # 状态转移矩阵（单位矩阵）
        self.F = np.eye(3)
        
        # 观测矩阵（单位矩阵）
        self.H = np.eye(3)
    
    def predict(self):
        """预测步骤"""
        # 状态预测
        self.state = self.F @ self.state
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 归一化状态向量
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    def update(self, measurement: np.ndarray):
        """
        更新步骤
        
        Args:
            measurement: 测量的重力向量
        """
        # 归一化测量向量
        measurement = measurement / np.linalg.norm(measurement)
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # 协方差更新
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        
        # 归一化状态向量
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        return self.state.copy()


class GravityEstimator:
    """重力向量估计器，使用多种平滑方法"""

    def __init__(self, method='exponential', window_size=10, alpha=0.1):
        """
        初始化重力向量估计器

        Args:
            method: 平滑方法，可选 'exponential'(指数移动平均), 
                    'moving'(移动平均), 'median'(中值滤波)
            window_size: 窗口大小（用于移动平均和中值滤波）
            alpha: 指数移动平均的平滑系数
        """
        self.method = method
        self.window_size = window_size
        self.alpha = alpha

        # 存储历史数据
        self.gravity_history = deque(maxlen=window_size)
        self.current_gravity = np.array([0.0, 0.0, -1.0])
        self.is_initialized = False
        
        # 卡尔曼滤波器
        if method == 'kalman':
            self.kalman_filter = KalmanFilterGravity()

    def update(self, new_gravity: np.ndarray) -> np.ndarray:
        """
        更新重力向量估计

        Args:
            new_gravity: 新的重力向量（3D向量）

        Returns:
            平滑后的重力向量
        """
        # 归一化输入向量
        new_gravity = new_gravity / np.linalg.norm(new_gravity)

        # 添加到历史记录
        self.gravity_history.append(new_gravity)

        # 根据方法进行平滑
        if self.method == 'kalman':
            self._kalman_filter_update(new_gravity)
        elif self.method == 'exponential':
            self.current_gravity = self._exponential_smoothing(new_gravity)
        elif self.method == 'moving':
            self.current_gravity = self._moving_average()
        elif self.method == 'median':
            self.current_gravity = self._median_filter()
        else:
            # 默认使用卡尔曼滤波
            self._kalman_filter_update(new_gravity)

        self.is_initialized = True
        return self.current_gravity

    def _kalman_filter_update(self, new_gravity: np.ndarray):
        """卡尔曼滤波更新"""
        self.kalman_filter.predict()
        self.kalman_filter.update(new_gravity)
        self.current_gravity = self.kalman_filter.get_state()

    def _exponential_smoothing(self, new_gravity: np.ndarray) -> np.ndarray:
        """指数移动平均"""
        if not self.is_initialized:
            return new_gravity
        return (1 - self.alpha) * self.current_gravity + self.alpha * new_gravity

    def _moving_average(self) -> np.ndarray:
        """移动平均"""
        if len(self.gravity_history) == 0:
            return self.current_gravity
        return np.mean(self.gravity_history, axis=0)

    def _median_filter(self) -> np.ndarray:
        """中值滤波"""
        if len(self.gravity_history) == 0:
            return self.current_gravity
        return np.median(self.gravity_history, axis=0)

    def get_current(self) -> np.ndarray:
        """获取当前重力向量"""
        return self.current_gravity.copy()

    def reset(self):
        """重置估计器"""
        self.gravity_history.clear()
        self.current_gravity = np.array([0.0, 0.0, -1.0])
        self.is_initialized = False
        if self.method == 'kalman':
            self.kalman_filter = KalmanFilterGravity()


class GravityEstimatorIMU(GravityEstimator):
    """基于IMU数据的重力向量估计器"""

    def __init__(self, method='kalman', window_size=10, alpha=0.1):
        """
        初始化基于IMU的重力向量估计器

        Args:
            method: 平滑方法
            window_size: 窗口大小
            alpha: 指数移动平均的平滑系数
        """
        super().__init__(method, window_size, alpha)
        self.bias = np.array([0.0, 0.0, 0.0])
        self.bias_samples = 0
        self.is_calibrated = False

    def update_from_imu(self, accel: np.ndarray, gyro: Optional[np.ndarray] = None) -> np.ndarray:
        """
        从IMU加速度数据更新重力向量

        Args:
            accel: IMU加速度数据（3D向量）
            gyro: IMU角速度数据（可选，用于检测运动状态）

        Returns:
            平滑后的重力向量
        """
        # 如果提供了角速度，检查是否静止
        if gyro is not None:
            gyro_norm = np.linalg.norm(gyro)
            # 如果角速度较大，说明在运动中，不更新重力估计
            if gyro_norm > 0.1:  # 阈值可调
                return self.current_gravity
        
        # 去除偏差
        accel_corrected = accel - self.bias
        
        # 归一化加速度向量作为重力估计
        accel_norm = accel_corrected / np.linalg.norm(accel_corrected)
        
        # 更新重力估计
        return self.update(accel_norm)

    def calibrate_bias(self, accel_samples: list):
        """
        标定IMU偏差

        Args:
            accel_samples: 静止状态下的加速度样本列表
        """
        if len(accel_samples) == 0:
            return

        # 计算平均偏差
        mean_accel = np.mean(accel_samples, axis=0)
        expected_gravity = np.array([0.0, 0.0, -9.81])
        self.bias = mean_accel - expected_gravity
        self.bias_samples = len(accel_samples)
        self.is_calibrated = True
        
        print(f"IMU bias calibrated: {self.bias}")
