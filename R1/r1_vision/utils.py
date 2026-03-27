"""
通用工具函数
"""
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional, Tuple

from scipy.spatial.transform import Rotation


def load_yaml_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载YAML配置文件
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        配置字典或None
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f'YAML解析失败: {e}')
        return None
    except Exception as e:
        print(f'配置加载失败: {e}')
        return None


def load_camera_calibration(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    从配置中提取相机内参
    
    Args:
        config: 配置字典
        
    Returns:
        相机内参字典 {'fx', 'fy', 'cx', 'cy'} 或None
    """
    if not config or 'camera_matrix' not in config:
        return None
    
    try:
        cm = np.array(config['camera_matrix'])
        return {
            'fx': float(cm[0, 0]),
            'fy': float(cm[1, 1]),
            'cx': float(cm[0, 2]),
            'cy': float(cm[1, 2])
        }
    except Exception as e:
        print(f'相机内参解析失败: {e}')
        return None


def validate_camera_params(fx: float, fy: float, cx: float, cy: float) -> bool:
    """
    验证相机内参是否合理
    
    Args:
        fx, fy, cx, cy: 相机内参
        
    Returns:
        是否有效
    """
    return (fx > 0 and fy > 0 and 
            cx > 0 and cy > 0 and
            fx < 2000 and fy < 2000 and  # 合理的焦距范围
            cx < 2000 and cy < 2000)     # 合理的主点范围


def setup_device(use_gpu: bool = True) -> str:
    """
    设置计算设备
    
    Args:
        use_gpu: 是否使用GPU
        
    Returns:
        设备字符串 ('cuda' 或 'cpu')
    """
    import torch
    import platform
    
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f'✅ GPU加速启用: {torch.cuda.get_device_name()}')
        print(f'CUDA版本: {torch.version.cuda}, PyTorch版本: {torch.__version__}')
    else:
        device = 'cpu'
        if use_gpu:
            print('⚠️ GPU不可用，使用CPU')
            if platform.machine() == 'aarch64':
                missing = [p for p in ['/dev/nvmap', '/dev/nvhost-ctrl', '/dev/nvhost-gpu'] if not os.path.exists(p)]
                if missing:
                    print(f'⚠️ Jetson GPU设备节点缺失: {missing}')
                    print('请在宿主机终端检查驱动与设备映射：ls -l /dev/nvmap /dev/nvhost-ctrl /dev/nvhost-gpu')
                    print('若在容器/沙箱中运行，请使用带NVIDIA设备映射的会话运行ROS2节点')
                    print('推荐在宿主机执行: source /opt/ros/humble/setup.bash && source ~/ws_livox/install/setup.bash')
        else:
            print('📱 使用CPU模式')
    
    return device


def cleanup_gpu():
    """清理GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class PerformanceMonitor:
    """性能监控器（推理耗时、回调处理耗时、消息到达延迟）"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化性能监控器
        
        Args:
            window_size: 滑动窗口大小
        """
        from collections import deque
        self._window = window_size
        self.inference_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.callback_latency_ms = deque(maxlen=window_size)
    
    def add_inference_time(self, time_ms: float):
        """添加推理时间"""
        self.inference_times.append(time_ms)
    
    def add_processing_time(self, time_ms: float):
        """添加处理时间"""
        self.processing_times.append(time_ms)

    def add_callback_latency_ms(self, latency_ms: float):
        """同步消息从采集时刻到本回调开始处理的大致延迟（毫秒）"""
        if np.isfinite(latency_ms) and latency_ms >= 0.0:
            self.callback_latency_ms.append(latency_ms)
    
    def get_avg_inference_time(self) -> float:
        """获取平均推理时间"""
        return float(np.mean(self.inference_times)) if self.inference_times else 0.0
    
    def get_avg_processing_time(self) -> float:
        """获取平均处理时间"""
        return float(np.mean(self.processing_times)) if self.processing_times else 0.0

    def get_avg_callback_latency_ms(self) -> float:
        return float(np.mean(self.callback_latency_ms)) if self.callback_latency_ms else 0.0

    def get_max_callback_latency_ms(self) -> float:
        return float(np.max(self.callback_latency_ms)) if self.callback_latency_ms else 0.0
    
    def get_fps(self) -> float:
        """获取FPS"""
        avg_time = self.get_avg_processing_time()
        return 1000.0 / avg_time if avg_time > 0 else 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'avg_inference_ms': self.get_avg_inference_time(),
            'avg_processing_ms': self.get_avg_processing_time(),
            'avg_callback_latency_ms': self.get_avg_callback_latency_ms(),
            'max_callback_latency_ms': self.get_max_callback_latency_ms(),
            'fps': self.get_fps(),
            'sample_count': len(self.processing_times)
        }


def format_position(position: np.ndarray, precision: int = 3) -> str:
    """
    格式化3D位置为字符串
    
    Args:
        position: 3D位置 [x, y, z]
        precision: 小数位数
        
    Returns:
        格式化字符串
    """
    return f"({position[0]:.{precision}f}, {position[1]:.{precision}f}, {position[2]:.{precision}f})"


def resolve_r1_config_path(explicit_path: str) -> Optional[str]:
    """
    解析 R1 集成/TF 配置文件路径。
    explicit_path 非空且存在则用之；否则尝试 ament share 下 config/tf_config.yaml；
    再尝试源码旁 config（未 install 时开发用）。
    """
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg = get_package_share_directory('r1_vision')
        p = os.path.join(pkg, 'config', 'tf_config.yaml')
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, '..', 'config', 'tf_config.yaml')
    dev = os.path.normpath(dev)
    if os.path.isfile(dev):
        return dev
    return None


def parse_lidar_to_camera_transform(cfg: Dict[str, Any]) -> np.ndarray:
    """
    从配置字典构建 4x4 齐次矩阵：p_cam = T @ p_lidar（列向量齐次）。

    支持：
    - matrix: 4x4 嵌套列表
    - translation: [tx,ty,tz] + rotation_rpy: [roll,pitch,yaw]（弧度，xyz 顺序）
    - translation + quaternion: [qx,qy,qz,qw]（与 scipy 一致）
    """
    if not cfg:
        return np.eye(4)

    if 'matrix' in cfg and cfg['matrix'] is not None:
        M = np.array(cfg['matrix'], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError('lidar_to_camera.matrix must be 4x4')
        # 兼容：若同时给了 translation，则用其覆盖矩阵平移列（便于现场只改位移）
        if 'translation' in cfg and cfg['translation'] is not None:
            t = np.array(cfg.get('translation', [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            M = M.copy()
            M[:3, 3] = t
        return M

    t = np.array(cfg.get('translation', [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    R = np.eye(3)
    if 'quaternion' in cfg and cfg['quaternion'] is not None:
        q = np.array(cfg['quaternion'], dtype=np.float64).reshape(4)
        R = Rotation.from_quat(q).as_matrix()
    elif 'rotation_rpy' in cfg and cfg['rotation_rpy'] is not None:
        rpy = np.array(cfg['rotation_rpy'], dtype=np.float64).reshape(3)
        R = Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_lidar_to_camera_from_config_file(path: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    从 YAML 文件读取 lidar_to_camera 变换。返回 (4x4, 完整 yaml dict 或 None)。
    """
    data = load_yaml_config(path)
    if not data:
        return np.eye(4), None
    lc = data.get('lidar_to_camera')
    if lc is None:
        return np.eye(4), data
    return parse_lidar_to_camera_transform(lc), data


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    限制数值范围
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_val, min(max_val, value))
