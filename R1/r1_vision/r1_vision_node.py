"""
R1 Vision ROS2节点 - 模块化视觉检测系统

集成所有R1模块的ROS2节点，提供：
- YOLO目标检测
- 深度图像处理
- Kalman滤波跟踪
- 法向量估计
- 抓取位姿计算
- 串口通信
"""
import time
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import message_filters
import tf2_ros
from threading import Lock
from typing import Optional

try:
    from livox_ros_driver2.msg import CustomMsg
    HAS_LIVOX_MSG = True
except ImportError:
    HAS_LIVOX_MSG = False

# 导入R1模块
from .detection import ObjectDetector
from .roi_processing import find_pattern_center, calculate_gray_variance
from .normal_estimation import (
    orient_normal_inward,
    validate_normal_with_gravity,
    NormalEstimatorWithSmoothing,
)
from .depth_processing import (
    get_median_depth,
    depth_roi_filtering,
)
from .normal_estimation import (
    estimate_normal_from_depth,
    estimate_normal_simple,
)
from .pose_estimation import compute_grasp_position_improved, validate_position, estimate_position_error
from .tracking import KalmanTracker
from .comms import SerialComm
from .gravity_estimation import GravityEstimatorIMU, GravityEstimator
from .pointcloud_fusion import PointCloudFusion
from .cube_localization import CubeLocalization
from rclpy.time import Time

from .utils import (
    setup_device,
    cleanup_gpu,
    PerformanceMonitor,
    format_position,
    resolve_r1_config_path,
    load_lidar_to_camera_from_config_file,
)


class R1VisionNode(Node):
    """R1 Vision ROS2节点 - 模块化、高性能视觉检测"""
    
    # 常量定义
    DEPTH_MIN = 0.1
    DEPTH_MAX = 10.0
    TRACKING_DISTANCE_THRESHOLD = 0.3
    
    def __init__(self):
        super().__init__('r1_vision_node')
        
        # 参数声明
        self._declare_parameters()
        
        # 获取参数
        params = self._get_parameters()
        self.params = params
        
        # 初始化组件
        self.bridge = CvBridge()
        self.performance_monitor = PerformanceMonitor()
        
        # 设备和模型初始化
        device = setup_device(params['use_gpu'])
        try:
            self.detector = ObjectDetector(params['model_path'], device)
            self.get_logger().info(f'✅ 检测器初始化成功: {params["model_path"]}')
        except Exception as e:
            self.get_logger().error(f'❌ 检测器初始化失败: {e}')
            return
        
        # 串口通信
        self.serial_comm = SerialComm(
            params['serial_port'], 
            params['baud_rate'], 
            params['mock_serial']
        )
        
        # 相机内参（硬编码）
        self.camera_params = {
            'fx': 926.599, 'fy': 926.238,
            'cx': 635.031, 'cy': 361.721
        }
        self.camera_info_received = True
        self.get_logger().info(
            f'✅ 使用硬编码相机内参: fx={self.camera_params["fx"]:.3f}, '
            f'fy={self.camera_params["fy"]:.3f}, cx={self.camera_params["cx"]:.3f}, cy={self.camera_params["cy"]:.3f}'
        )
        
        # 跟踪器
        self.tracker: Optional[KalmanTracker] = None
        self.tracker_lock = Lock()
        self.use_kalman = params['use_kalman']
        
        # 算法参数
        self.depth_window = params['depth_window']
        self.normal_window = params['normal_window']
        self.enable_light_adaptation = params['enable_light_adaptation']
        self.variance_threshold = params['variance_threshold']
        self.frustum_bbox_shrink_ratio = float(params['frustum_bbox_shrink_ratio'])
        self.frustum_min_points = int(params['frustum_min_points'])
        self.frustum_min_cube_points = int(params['frustum_min_cube_points'])
        self.frustum_foreground_depth_range = float(params['frustum_foreground_depth_range'])
        self.frustum_foreground_depth_percentile = float(params['frustum_foreground_depth_percentile'])
        self.enable_grasp_consistency_gate = bool(params['grasp_consistency_enable'])
        self.grasp_max_jump_m = float(params['grasp_max_jump_m'])
        self.grasp_max_normal_angle_deg = float(params['grasp_max_normal_angle_deg'])
        self.grasp_confidence_threshold = float(params['grasp_confidence_threshold'])
        self.grasp_min_confirm_frames = max(1, int(params['grasp_min_confirm_frames']))
        self.camera_only_confidence = float(params['camera_only_confidence'])
        self._stable_grasp_counter = 0
        self._last_sent_grasp = None
        self._last_sent_normal = None
        
        # 性能控制
        self.frame_skip_count = 0
        self._base_process_every_n_frames = max(1, int(params.get('process_every_n_frames', 2)))
        self.process_every_n_frames = self._base_process_every_n_frames
        self.adaptive_frame_skip = params.get('adaptive_frame_skip', True)
        self.target_processing_ms = params.get('target_processing_ms', 100.0)
        self.max_process_every_n_frames = max(
            self._base_process_every_n_frames,
            int(params.get('max_process_every_n_frames', 8)),
        )
        self.adaptive_adjust_interval = max(1, int(params.get('adaptive_adjust_interval', 30)))
        self._adaptive_skip_counter = 0
        
        # 日志控制
        self.log_counter = 0
        self.log_every_n_sends = params['log_every_n_sends']
        
        # QoS设置
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 定义回调组：主流程同步回调使用独立组，避免被IMU/Odometry高频数据阻塞
        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        
        # ROS订阅（同步）
        self._setup_subscriptions(qos)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 重力向量相关
        self.current_gravity_vector_base = np.array([0.0, 0.0, -1.0])
        self.current_gravity_vector_cam = np.array([0.0, 0.0, -1.0])
        self.gravity_lock = Lock()
        
        # 重力估计器（使用配置参数）
        self.gravity_estimator = GravityEstimator(
            method=params['gravity_method'],
            window_size=params['gravity_window_size'],
            alpha=params['gravity_alpha']
        )
        self.imu_gravity_estimator = GravityEstimatorIMU(
            method=params['gravity_method'],
            window_size=params['gravity_window_size'],
            alpha=params['gravity_alpha']
        )
        self.use_imu_gravity = params['use_imu_gravity']  # 优先使用IMU重力
        
        # 订阅IMU数据 (mid360)
        self.imu_sub = self.create_subscription(
            Imu, self.params['topic_imu'], self.imu_callback,
            10, callback_group=self.sensor_cb_group
        )
        
        # 点云融合器（雷达到相机外参从 tf_config.yaml 的 lidar_to_camera 读取）
        self.pointcloud_fusion = PointCloudFusion()
        cfg_path = resolve_r1_config_path((params.get('r1_config_file') or '').strip())
        if cfg_path:
            try:
                T_lc, _ = load_lidar_to_camera_from_config_file(cfg_path)
                self.pointcloud_fusion.set_transform(T_lc)
                self.get_logger().info(f'已加载雷达到相机外参: {cfg_path}')
            except Exception as e:
                self.get_logger().warn(f'解析 lidar_to_camera 失败，使用单位阵: {e}')
        else:
            self.get_logger().warn(
                '未找到 R1 配置文件（tf_config.yaml），lidar_to_camera 使用单位阵；'
                '请设置参数 r1_config_file 或安装包内 config'
            )
        self.use_pointcloud_fusion = params['use_pointcloud_fusion']  # 是否使用点云融合
        self.fusion_adaptive_enable = bool(params['fusion_adaptive_enable'])
        self.fusion_disable_processing_ms = float(params['fusion_disable_processing_ms'])
        self.fusion_enable_processing_ms = float(params['fusion_enable_processing_ms'])
        self._fusion_runtime_enabled = bool(self.use_pointcloud_fusion)
        self.fusion_max_distance = float(params['fusion_max_distance'])
        self.fusion_min_confidence = float(params['fusion_min_confidence'])
        self.fusion_depth_weight_scale = float(params['fusion_depth_weight_scale'])
        self.target_output_frame = str(params['target_output_frame']).strip() or 'base_link'
        
        self.cube_side_length = params['cube_side_length']
        # 立方体融合定位器（方案3：相机+雷达融合）
        self.cube_localization = CubeLocalization(cube_side_length=self.cube_side_length)
        self.use_fusion_localization = params['use_fusion_localization']
        
        # 法向量估计器（带平滑，使用配置参数）
        self.normal_estimator = NormalEstimatorWithSmoothing(
            window_size=params['normal_window_size'],
            expect_side_face=params['normal_expect_side_face'],
            max_interframe_angle_deg=params['normal_max_interframe_angle_deg'],
        )
        self.use_normal_smoothing = params['use_normal_smoothing']  # 是否使用法向量平滑
        self.normal_angle_threshold = float(params['normal_angle_threshold'])
        self.normal_expect_side_face = params['normal_expect_side_face']

        gf = np.array(params['gravity_fallback_cam'], dtype=np.float64)
        gn = np.linalg.norm(gf)
        self._gravity_fallback_cam = gf / gn if gn > 1e-9 else np.array([0.0, 0.0, -1.0])

        # 订阅FAST-LIO的里程计信息
        self.odom_sub = self.create_subscription(
            Odometry, self.params['topic_odom'], self.odom_callback,
            10, callback_group=self.sensor_cb_group
        )

        # 定时更新重力向量
        self.gravity_update_timer = self.create_timer(0.1, self.update_gravity_vector, callback_group=self.timer_cb_group)
        
        # 增加看门狗定时器，检测数据是否丢失
        self.last_sync_time = time.time()
        self.watchdog_timer = self.create_timer(2.0, self.watchdog_callback, callback_group=self.timer_cb_group)
        
        # RViz可视化发布器
        self.normal_marker_pub = self.create_publisher(Marker, '/r1_vision/normal_marker', 10)
        self.grasp_marker_pub = self.create_publisher(Marker, '/r1_vision/grasp_marker', 10)
        self.gravity_marker_pub = self.create_publisher(Marker, '/r1_vision/gravity_marker', 10)
        
        self.get_logger().info('🚀 R1 Vision节点启动完成')
        self.get_logger().info(f'✅ 模块化架构 | GPU: {device} | Kalman: {self.use_kalman}')
    
    def _compute_3d_position(self, rgb_image: np.ndarray, depth_image: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """基于深度图与相机内参估计3D位置（ROI中值与平面拟合结合）"""
        try:
            bbox = detection['bbox']
            cx, cy = find_pattern_center(rgb_image, bbox)
            pos = depth_roi_filtering(
                depth_image, np.array(bbox),
                center_x=cx, center_y=cy,
                fx=self.camera_params['fx'],
                fy=self.camera_params['fy'],
                cx=self.camera_params['cx'],
                cy=self.camera_params['cy'],
            )
            if pos is None:
                d = get_median_depth(depth_image, cx, cy, self.depth_window)
                if d is None or d <= self.DEPTH_MIN:
                    return None
                X = (cx - self.camera_params['cx']) * d / self.camera_params['fx']
                Y = (cy - self.camera_params['cy']) * d / self.camera_params['fy']
                Z = d
                return np.array([X, Y, Z], dtype=np.float32)
            return pos.astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'3D位置估计失败: {e}')
            return None
    
    def _declare_parameters(self):
        """声明ROS参数"""
        # 话题名参数（从硬编码改为可配置）
        self.declare_parameter('topic_rgb', '/camera/camera/color/image_raw')
        self.declare_parameter('topic_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('topic_lidar_points', '/livox/lidar')   # mid360
        self.declare_parameter('topic_imu', '/livox/imu')              # mid360
        self.declare_parameter('topic_odom', '/Odometry')
        # 同步参数
        self.declare_parameter('sync_slop', 0.03)
        # 模型与串口
        self.declare_parameter('model_path', '$(find-pkg-share r1_vision)/models/best.pt')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('depth_window', 5)
        self.declare_parameter('normal_window', 7)
        self.declare_parameter('enable_light_adaptation', True)
        self.declare_parameter('variance_threshold', 100.0)
        self.declare_parameter('use_kalman', True)
        self.declare_parameter('mock_serial', False)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('log_level', 'INFO')
        self.declare_parameter('process_every_n_frames', 2)
        self.declare_parameter('log_every_n_sends', 10)
        # 重力估计参数
        self.declare_parameter('use_imu_gravity', True)
        self.declare_parameter('gravity_method', 'kalman')  # kalman, exponential, moving, median
        self.declare_parameter('gravity_window_size', 10)
        self.declare_parameter('gravity_alpha', 0.1)
        # 点云融合参数
        self.declare_parameter('use_pointcloud_fusion', True)
        self.declare_parameter('fusion_adaptive_enable', True)
        self.declare_parameter('fusion_disable_processing_ms', 140.0)
        self.declare_parameter('fusion_enable_processing_ms', 95.0)
        self.declare_parameter('fusion_max_distance', 0.05)
        self.declare_parameter('fusion_min_confidence', 0.3)
        # 法向量估计参数
        self.declare_parameter('use_normal_smoothing', True)
        self.declare_parameter('normal_window_size', 5)
        self.declare_parameter('normal_angle_threshold', 45.0)
        # 最终下发坐标系（默认 base_link，可按需改为 map/odom 等）
        self.declare_parameter('target_output_frame', 'base_link')
        # TF 失败或重力未就绪时，相机系下的重力方向（单位向量近似，将归一化）
        self.declare_parameter('gravity_fallback_cam', [0.0, 0.0, -1.0])
        self.declare_parameter('use_fusion_localization', True)
        self.declare_parameter('cube_side_length', 0.35)
        self.declare_parameter('normal_expect_side_face', True)
        # 配置文件：默认使用包内 share/r1_vision/config/tf_config.yaml（含 lidar_to_camera）
        self.declare_parameter('r1_config_file', '$(find-pkg-share r1_vision)/config/tf_config.yaml')
        # 负载：自适应跳帧（在 process_every_n_frames 基础上动态增减）
        self.declare_parameter('adaptive_frame_skip', True)
        self.declare_parameter('target_processing_ms', 100.0)
        self.declare_parameter('max_process_every_n_frames', 8)
        self.declare_parameter('adaptive_adjust_interval', 30)
        # 融合：深度加权（0=关闭，0.3~1.0 常见）
        self.declare_parameter('fusion_depth_weight_scale', 0.5)
        # 法向：帧间夹角超过则重置平滑窗口（0=关闭）
        self.declare_parameter('normal_max_interframe_angle_deg', 40.0)
        self.declare_parameter('frustum_bbox_shrink_ratio', 0.1)
        self.declare_parameter('frustum_min_points', 15)
        self.declare_parameter('frustum_min_cube_points', 15)
        self.declare_parameter('frustum_foreground_depth_range', 0.3)
        self.declare_parameter('frustum_foreground_depth_percentile', 35.0)
        # 发送稳定性增强参数（动态场景）
        self.declare_parameter('grasp_consistency_enable', True)
        self.declare_parameter('grasp_max_jump_m', 0.12)
        self.declare_parameter('grasp_max_normal_angle_deg', 30.0)
        self.declare_parameter('grasp_confidence_threshold', 0.60)
        self.declare_parameter('grasp_min_confirm_frames', 2)
        self.declare_parameter('camera_only_confidence', 0.55)
        # TF缺失时是否降级发送camera_link坐标（而不丢弃）
        self.declare_parameter('fallback_send_without_tf', True)
        self.declare_parameter('lidar_msg_type', 'custom') # 'custom' for Livox CustomMsg, 'pointcloud2' for standard PointCloud2
    
    def _get_parameters(self) -> dict:
        """获取参数值"""
        return {
            # 话题名参数
            'topic_rgb': self.get_parameter('topic_rgb').value,
            'topic_depth': self.get_parameter('topic_depth').value,
            'topic_lidar_points': self.get_parameter('topic_lidar_points').value,
            'topic_imu': self.get_parameter('topic_imu').value,
            'topic_odom': self.get_parameter('topic_odom').value,
            'sync_slop': float(self.get_parameter('sync_slop').value),
            # 模型与串口
            'model_path': self.get_parameter('model_path').value,
            'serial_port': self.get_parameter('serial_port').value,
            'baud_rate': self.get_parameter('baud_rate').value,
            'depth_window': self.get_parameter('depth_window').value,
            'normal_window': self.get_parameter('normal_window').value,
            'enable_light_adaptation': self.get_parameter('enable_light_adaptation').value,
            'variance_threshold': self.get_parameter('variance_threshold').value,
            'use_kalman': self.get_parameter('use_kalman').value,
            'mock_serial': self.get_parameter('mock_serial').value,
            'use_gpu': self.get_parameter('use_gpu').value,
            'log_level': self.get_parameter('log_level').value,
            'process_every_n_frames': self.get_parameter('process_every_n_frames').value,
            'log_every_n_sends': self.get_parameter('log_every_n_sends').value,
            # 重力估计参数
            'use_imu_gravity': self.get_parameter('use_imu_gravity').value,
            'gravity_method': self.get_parameter('gravity_method').value,
            'gravity_window_size': self.get_parameter('gravity_window_size').value,
            'gravity_alpha': self.get_parameter('gravity_alpha').value,
            # 点云融合参数
            'use_pointcloud_fusion': self.get_parameter('use_pointcloud_fusion').value,
            'fusion_adaptive_enable': self.get_parameter('fusion_adaptive_enable').value,
            'fusion_disable_processing_ms': float(self.get_parameter('fusion_disable_processing_ms').value),
            'fusion_enable_processing_ms': float(self.get_parameter('fusion_enable_processing_ms').value),
            'fusion_max_distance': self.get_parameter('fusion_max_distance').value,
            'fusion_min_confidence': self.get_parameter('fusion_min_confidence').value,
            # 法向量估计参数
            'use_normal_smoothing': self.get_parameter('use_normal_smoothing').value,
            'normal_window_size': self.get_parameter('normal_window_size').value,
            'normal_angle_threshold': self.get_parameter('normal_angle_threshold').value,
            'target_output_frame': self.get_parameter('target_output_frame').value,
            'gravity_fallback_cam': list(self.get_parameter('gravity_fallback_cam').value),
            'use_fusion_localization': self.get_parameter('use_fusion_localization').value,
            'cube_side_length': float(self.get_parameter('cube_side_length').value),
            'normal_expect_side_face': self.get_parameter('normal_expect_side_face').value,
            'r1_config_file': self.get_parameter('r1_config_file').value,
            'adaptive_frame_skip': self.get_parameter('adaptive_frame_skip').value,
            'target_processing_ms': float(self.get_parameter('target_processing_ms').value),
            'max_process_every_n_frames': int(self.get_parameter('max_process_every_n_frames').value),
            'adaptive_adjust_interval': int(self.get_parameter('adaptive_adjust_interval').value),
            'fusion_depth_weight_scale': float(self.get_parameter('fusion_depth_weight_scale').value),
            'normal_max_interframe_angle_deg': float(
                self.get_parameter('normal_max_interframe_angle_deg').value
            ),
            'frustum_bbox_shrink_ratio': float(self.get_parameter('frustum_bbox_shrink_ratio').value),
            'frustum_min_points': int(self.get_parameter('frustum_min_points').value),
            'frustum_min_cube_points': int(self.get_parameter('frustum_min_cube_points').value),
            'frustum_foreground_depth_range': float(
                self.get_parameter('frustum_foreground_depth_range').value
            ),
            'frustum_foreground_depth_percentile': float(
                self.get_parameter('frustum_foreground_depth_percentile').value
            ),
            'grasp_consistency_enable': self.get_parameter('grasp_consistency_enable').value,
            'grasp_max_jump_m': float(self.get_parameter('grasp_max_jump_m').value),
            'grasp_max_normal_angle_deg': float(self.get_parameter('grasp_max_normal_angle_deg').value),
            'grasp_confidence_threshold': float(self.get_parameter('grasp_confidence_threshold').value),
            'grasp_min_confirm_frames': int(self.get_parameter('grasp_min_confirm_frames').value),
            'camera_only_confidence': float(self.get_parameter('camera_only_confidence').value),
            'fallback_send_without_tf': self.get_parameter('fallback_send_without_tf').value in (True, 'true', 'True', '1', 1),
            'lidar_msg_type': self.get_parameter('lidar_msg_type').value,
        }
    
    def _setup_subscriptions(self, qos):
        """设置ROS订阅（只订阅必需的话题，不再订阅相机点云和深度图）"""
        p = self.params
        
        rgb_sub = message_filters.Subscriber(self, Image, p['topic_rgb'], qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, p['topic_depth'], qos_profile=qos)
        
        if p['lidar_msg_type'] == 'custom' and HAS_LIVOX_MSG:
            lidar_pc_sub = message_filters.Subscriber(self, CustomMsg, p['topic_lidar_points'], qos_profile=qos)
            self.get_logger().info("使用 Livox CustomMsg 订阅雷达点云")
        else:
            lidar_pc_sub = message_filters.Subscriber(self, PointCloud2, p['topic_lidar_points'], qos_profile=qos)
            self.get_logger().info("使用 PointCloud2 订阅雷达点云")
            
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, lidar_pc_sub], queue_size=10, slop=p['sync_slop']
        )
        self.ts.registerCallback(self.synced_callback)
        
        # 相机内参已硬编码，无需订阅
    
    def watchdog_callback(self):
        """看门狗，检查主回调是否卡死"""
        if time.time() - self.last_sync_time > 2.0:
            self.get_logger().warn('⚠️ 看门狗警告：超过 2 秒未收到同步传感器数据（可能雷达或相机掉线）')
            if self.use_kalman and self.tracker is not None:
                with self.tracker_lock:
                    self.tracker = None  # 数据丢失时重置跟踪器
            
    def imu_callback(self, msg: Imu):
        """
        处理来自雷达的IMU数据，更新重力向量估计。
        """
        if not self.use_imu_gravity:
            return
            
        try:
            # 提取加速度和角速度
            accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            # 使用IMU重力估计器更新
            with self.gravity_lock:
                gravity_imu = self.imu_gravity_estimator.update_from_imu(accel, gyro)
                # 更新base_link坐标系下的重力向量
                self.current_gravity_vector_base = gravity_imu
                
        except Exception as e:
            self.get_logger().warn(f'IMU数据处理失败: {e}')

    def odom_callback(self, msg: Odometry):
        """
        处理来自 FAST-LIO 的里程计数据，更新机器人的姿态。
        """
        if self.use_imu_gravity:
            return
        try:
            # 将四元数转换为旋转矩阵
            orientation = msg.pose.pose.orientation
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            
            from scipy.spatial.transform import Rotation as R
            rot_mat = R.from_quat(q).as_matrix()
            
            # 世界坐标系下的重力向量 (通常指向 Z 轴负方向)
            gravity_world = np.array([0.0, 0.0, -1.0])
            
            # 将世界坐标系下的重力向量旋转到机器人坐标系（base_link）下
            gravity_base = rot_mat.T @ gravity_world
            
            # 使用重力估计器进行平滑
            with self.gravity_lock:
                self.current_gravity_vector_base = self.gravity_estimator.update(gravity_base)
        except Exception as e:
            self.get_logger().warn(f'里程计数据处理失败: {e}')

    def update_gravity_vector(self):
        """
        将 base_link 下的重力向量转换到 camera_link 下。
        """
        try:
            # 获取从 base_link 到 camera_link 的变换（带超时；旧版 rclpy/tf2 无 timeout 参数则回退）
            try:
                transform = self.tf_buffer.lookup_transform(
                    'camera_link',
                    'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
            except TypeError:
                transform = self.tf_buffer.lookup_transform(
                    'camera_link',
                    'base_link',
                    rclpy.time.Time(),
                )
            
            # 将 TransformStamped 消息转换为旋转矩阵
            q = transform.transform.rotation
            from scipy.spatial.transform import Rotation as R_scipy
            rot_mat = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            
            # 将重力向量从 base_link 转换到 camera_link
            with self.gravity_lock:
                self.current_gravity_vector_cam = rot_mat @ self.current_gravity_vector_base
                
            self.get_logger().info(f'Updated gravity vector in camera frame: {self.current_gravity_vector_cam}', throttle_duration_sec=1.0)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}, using gravity_fallback_cam', throttle_duration_sec=1.0)
            with self.gravity_lock:
                self.current_gravity_vector_cam = self._gravity_fallback_cam.copy()

    def _publish_normal_marker(self, position: np.ndarray, normal: np.ndarray, timestamp):
        """发布法向量可视化标记"""
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = 'camera_link'
        marker.ns = 'normal'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.points = [
            Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            Point(x=float(position[0] + normal[0] * 0.15),
                  y=float(position[1] + normal[1] * 0.15),
                  z=float(position[2] + normal[2] * 0.15))
        ]
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.0
        
        self.normal_marker_pub.publish(marker)

    def _publish_grasp_marker(self, position: np.ndarray, timestamp):
        """发布抓取位置可视化标记"""
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = 'camera_link'
        marker.ns = 'grasp'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.grasp_marker_pub.publish(marker)

    def _publish_gravity_marker(self, timestamp):
        """发布重力向量可视化标记"""
        with self.gravity_lock:
            gravity = self.current_gravity_vector_cam.copy()
        
        # 从原点出发
        origin = np.array([0.0, 0.0, 0.0])
        
        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = 'camera_link'
        marker.ns = 'gravity'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.points = [
            Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            Point(x=float(gravity[0] * 0.2),
                  y=float(gravity[1] * 0.2),
                  z=float(gravity[2] * 0.2))
        ]
        
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.0
        
        self.gravity_marker_pub.publish(marker)


    
    def synced_callback(self, rgb_msg: Image, depth_msg: Image, lidar_pc_msg):
        """同步回调 - 主处理流程"""
        self.last_sync_time = time.time()
        try:
            now = self.get_clock().now()
            msg_time = Time.from_msg(rgb_msg.header.stamp)
            lat_ms = (now - msg_time).nanoseconds / 1e6
            if lat_ms < 0.0:
                lat_ms = 0.0
            self.performance_monitor.add_callback_latency_ms(lat_ms)
        except Exception:
            pass

        # 性能优化：跳帧处理
        self.frame_skip_count += 1
        if self.frame_skip_count % self.process_every_n_frames != 0:
            return
        
        start_time = time.time()

        self._update_fusion_runtime_switch()
        
        try:
            # 图像转换
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if isinstance(depth_raw, np.ndarray):
                if depth_raw.dtype == np.uint16:
                    depth_image = (depth_raw.astype(np.float32)) / 1000.0
                elif depth_raw.dtype in (np.float32, np.float64):
                    depth_image = depth_raw.astype(np.float32)
                else:
                    depth_image = depth_raw.astype(np.float32)
            else:
                depth_image = None
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return
        
        # 目标检测
        inference_start = time.time()
        detections = self.detector.detect(cv_image)
        inference_time = (time.time() - inference_start) * 1000
        self.performance_monitor.add_inference_time(inference_time)
        
        # 处理检测结果
        if not detections:
            self._handle_no_detection(rgb_msg.header.stamp)
            return
        
        # 选择最佳检测
        best_detection = self._select_best_detection(detections, cv_image)
        if best_detection is None:
            self._handle_no_detection(rgb_msg.header.stamp)
            return
            
        # 点云准备 (雷达点云)
        lidar_points_array = None
        try:
            if hasattr(lidar_pc_msg, 'points'):  # CustomMsg
                # 提取 x, y, z
                pts = [[p.x, p.y, p.z] for p in lidar_pc_msg.points]
                lidar_points_array = np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)
            else:  # PointCloud2
                import sensor_msgs_py.point_cloud2 as pc2
                # 转换雷达点云
                lidar_points = pc2.read_points(lidar_pc_msg, field_names=["x", "y", "z"], skip_nans=True)
                lidar_points_array = np.array(list(lidar_points), dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f'雷达点云处理失败: {e}')
            self._handle_no_detection(rgb_msg.header.stamp)
            return

        # 先做雷达点云预过滤，避免全量点进入融合/定位流程
        if lidar_points_array is not None and len(lidar_points_array) > 0:
            lidar_points_array = self._prefilter_lidar_points(lidar_points_array)
            if len(lidar_points_array) == 0:
                self.get_logger().warn('雷达预过滤后无有效点，跳过本帧')
                self._handle_no_detection(rgb_msg.header.stamp)
                return
        
        # 计算3D位置 (基于纯视觉兜底，不一定成功，但需要传入)
        position_3d = self._compute_3d_position(
            cv_image, depth_image, best_detection
        )
        
        # 计算抓取位置并发送（传入雷达点云用于融合定位，传入纯视觉 position_3d 用于兜底）
        self._compute_and_send_grasp(
            cv_image, depth_image, position_3d, best_detection, rgb_msg.header.stamp,
            lidar_points=lidar_points_array
        )
        
        # 性能统计
        processing_time = (time.time() - start_time) * 1000
        self.performance_monitor.add_processing_time(processing_time)

        if self.adaptive_frame_skip:
            self._adaptive_skip_counter += 1
            if self._adaptive_skip_counter >= self.adaptive_adjust_interval:
                self._adaptive_skip_counter = 0
                avg = self.performance_monitor.get_avg_processing_time()
                if avg > self.target_processing_ms * 1.15:
                    if self.process_every_n_frames < self.max_process_every_n_frames:
                        self.process_every_n_frames += 1
                        self.get_logger().info(
                            f'负载偏高 (avg {avg:.0f}ms)，跳帧增至每 {self.process_every_n_frames} 帧处理 1 次',
                            throttle_duration_sec=5.0,
                        )
                elif (
                    avg < self.target_processing_ms * 0.55
                    and self.process_every_n_frames > self._base_process_every_n_frames
                ):
                    self.process_every_n_frames -= 1
                    self.get_logger().info(
                        f'负载降低 (avg {avg:.0f}ms)，跳帧减至每 {self.process_every_n_frames} 帧处理 1 次',
                        throttle_duration_sec=5.0,
                    )
    
    def _select_best_detection(self, detections: list, image: np.ndarray) -> dict:
        """选择最佳检测"""
        if len(detections) == 1:
            return detections[0]
        
        if not self.enable_light_adaptation:
            # 选择置信度最高的
            return max(detections, key=lambda d: d['confidence'])
        
        # 光照自适应选择
        best_detection = None
        best_score = 0.0
        
        for detection in detections:
            confidence = detection['confidence']
            variance = calculate_gray_variance(image, detection['bbox'])
            
            # 综合评分
            normalized_var = min(variance / self.variance_threshold, 1.0)
            score = 0.7 * confidence + 0.3 * normalized_var
            
            if score > best_score and score > 0.3:
                best_score = score
                best_detection = detection
        
        return best_detection
    
    def _update_tracker(self, detection: dict, position_3d: np.ndarray):
        """更新跟踪器"""
        if not self.use_kalman:
            return
        
        with self.tracker_lock:
            confidence = detection['confidence']
            
            if self.tracker is not None and self.tracker.is_valid():
                predicted_pos = self.tracker.predict()
                distance = np.linalg.norm(predicted_pos - position_3d)
                
                if distance < self.TRACKING_DISTANCE_THRESHOLD:
                    self.tracker.update(position_3d, confidence)
                elif confidence > 0.8 and self.tracker.get_confidence() < 0.6:
                    # 替换跟踪器
                    self.tracker = KalmanTracker(position_3d)
                    self.tracker.update(position_3d, confidence)
                else:
                    self.tracker.missed_frames += 1
            else:
                # 创建新跟踪器
                self.tracker = KalmanTracker(position_3d)
                self.tracker.update(position_3d, confidence)
    
    def _get_final_position(self, detection_position: np.ndarray) -> np.ndarray:
        """获取最终位置（跟踪或检测）"""
        if self.use_kalman and self.tracker is not None:
            with self.tracker_lock:
                if self.tracker.is_valid():
                    return self.tracker.predict()
        return detection_position
    
    def _compute_and_send_grasp(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                               position_3d: np.ndarray, detection: dict, timestamp,
                               lidar_points: np.ndarray = None):
        """计算抓取位置并发送 (相机深度与雷达融合共同作用)
        
        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            position_3d: 纯视觉估计的当前3D位置
            detection: 检测结果
            timestamp: 时间戳
            lidar_points: 雷达点云（可选，用于融合定位）
        """
        bbox = detection['bbox']
        
        # 获取重力向量
        with self.gravity_lock:
            gravity_vector = self.current_gravity_vector_cam.copy()

        # ==================== 1. 提取相机深度信息 ====================
        camera_normal = None
        camera_valid = False
        
        if position_3d is not None:
            pattern_cx, pattern_cy = find_pattern_center(rgb_image, bbox)
            pattern_depth = get_median_depth(depth_image, pattern_cx, pattern_cy, self.depth_window)
            
            if pattern_depth is not None and pattern_depth > self.DEPTH_MIN:
                # 估计法向量
                camera_normal = estimate_normal_from_depth(
                    depth_image, pattern_cx, pattern_cy,
                    **self.camera_params, window=self.normal_window
                )
                
                if camera_normal is None:
                    camera_normal = estimate_normal_simple(
                        depth_image, pattern_cx, pattern_cy,
                        fx=self.camera_params['fx'],
                        fy=self.camera_params['fy'],
                        cx=self.camera_params['cx'],
                        cy=self.camera_params['cy'],
                    )
                
                if camera_normal is not None:
                    # 法向量指向内部
                    camera_normal = orient_normal_inward(position_3d, camera_normal)
                    camera_valid = True

        # ==================== 2. 点云融合（可选） ====================
        fused_lidar_points = lidar_points
        if self._fusion_runtime_enabled and lidar_points is not None and len(lidar_points) > 0 and position_3d is not None:
            try:
                # 从深度图生成相机点云（ROI区域）
                camera_points = self._extract_camera_points_from_depth(
                    depth_image, bbox, position_3d
                )
                if camera_points is not None and len(camera_points) > 0:
                    # 融合雷达和相机点云
                    fused_points, confidences = self.pointcloud_fusion.fuse_pointclouds(
                        lidar_points=lidar_points,
                        camera_points=camera_points,
                        max_distance=self.fusion_max_distance,
                        fusion_depth_weight_scale=self.fusion_depth_weight_scale,
                    )
                    # 根据置信度过滤
                    if len(fused_points) > 0:
                        fused_lidar_points = self.pointcloud_fusion.filter_by_confidence(
                            fused_points, confidences, min_confidence=self.fusion_min_confidence
                        )
                        self.get_logger().info(
                            f'🔄 点云融合: 雷达 {len(lidar_points)} 点 + 相机 {len(camera_points)} 点 '
                            f'→ 融合后 {len(fused_lidar_points)} 点'
                        )
            except Exception as e:
                self.get_logger().warn(f'点云融合失败，使用原始雷达点云: {e}')
                fused_lidar_points = lidar_points
        
        # ==================== 3. 提取雷达 Frustum 融合信息 ====================
        lidar_valid = False
        lidar_center = None
        lidar_normal = None
        lidar_confidence = 0.0
        
        if self.use_fusion_localization and fused_lidar_points is not None and len(fused_lidar_points) > 0:
            fusion_result = self.cube_localization.localize_from_frustum_and_pointcloud(
                bbox=bbox,
                camera_params=self.camera_params,
                lidar_points=fused_lidar_points,  # 使用融合后的点云
                lidar_to_camera_transform=self.pointcloud_fusion.lidar_to_camera_transform,
                bbox_shrink_ratio=self.frustum_bbox_shrink_ratio,
                min_frustum_points=self.frustum_min_points,
                min_cube_points=self.frustum_min_cube_points,
                foreground_depth_range=self.frustum_foreground_depth_range,
                foreground_depth_percentile=self.frustum_foreground_depth_percentile,
            )
            
            if fusion_result is not None and fusion_result['confidence'] > 0.5:
                lidar_center = fusion_result['side_center']
                lidar_normal = fusion_result['normal']
                lidar_confidence = fusion_result['confidence']
                lidar_valid = True

        # ==================== 5. 结果合并 (Blending) ====================
        final_center = None
        final_normal = None
        result_confidence = 0.0
        
        if lidar_valid and camera_valid:
            # 都有数据时进行加权融合
            # 雷达对距离非常准，相机对局部法向量更敏感，这里我们可以根据权重进行合并
            weight_lidar = 0.7  # 雷达权重更高
            weight_camera = 1.0 - weight_lidar
            
            # 合并中心点
            final_center = weight_lidar * lidar_center + weight_camera * position_3d
            
            # 合并法向量 (需要归一化)
            blended_normal = weight_lidar * lidar_normal + weight_camera * camera_normal
            final_normal = blended_normal / np.linalg.norm(blended_normal)
            result_confidence = float(np.clip(0.5 * lidar_confidence + 0.5, 0.0, 1.0))
            
            self.get_logger().info(f'🔄 雷达与深度图双重融合: 雷达置信度 {lidar_confidence:.2f}')
            
        elif lidar_valid:
            final_center = lidar_center
            final_normal = lidar_normal
            result_confidence = float(np.clip(lidar_confidence, 0.0, 1.0))
            self.get_logger().info(f'🎯 仅雷达有效: 雷达置信度 {lidar_confidence:.2f}')
            
        elif camera_valid:
            final_center = position_3d
            final_normal = camera_normal
            result_confidence = float(np.clip(self.camera_only_confidence, 0.0, 1.0))
            self.get_logger().info('⚠️ 仅深度图有效，使用纯视觉兜底')
            
        else:
            self.get_logger().warn('❌ 雷达与深度图均失效，无法定位')
            self._handle_no_detection(timestamp)
            return

        # ==================== 4. 跟踪、平滑与抓取计算 ====================
        
        # 使用法向量估计器进行平滑和验证
        if self.use_normal_smoothing:
            normal_smoothed, _ = self.normal_estimator.update(final_normal, gravity_vector)
            
            is_valid, angle = validate_normal_with_gravity(
                normal_smoothed,
                gravity_vector,
                angle_threshold=self.normal_angle_threshold,
                expect_side_face=self.normal_expect_side_face,
            )
            
            if not is_valid:
                self.get_logger().warn(
                    f'法向量与重力关系异常（侧面模式夹角≈{angle:.1f}°），回退到未平滑法向量'
                )
            else:
                final_normal = normal_smoothed
        
        # 更新跟踪器 (跟踪 final_center)
        self._update_tracker(detection, final_center)
        tracked_center = self._get_final_position(final_center)
        
        # 用平滑后的 normal 和 跟踪后的 center 计算抓取位置
        grasp_position = compute_grasp_position_improved(
            tracked_center, final_normal,
            cube_side_length=self.cube_side_length,
            gravity=gravity_vector,
        )
        
        # 发布可视化
        self._publish_normal_marker(tracked_center, final_normal, timestamp)
        self._publish_grasp_marker(grasp_position, timestamp)
        self._publish_gravity_marker(timestamp)
        
        # 误差估计
        error_info = estimate_position_error(
            tracked_center, final_normal, gravity_vector, cube_side_length=self.cube_side_length
        )
        
        self.get_logger().info(
            f'双重融合 - 侧面中心: [{tracked_center[0]:.3f}, {tracked_center[1]:.3f}, {tracked_center[2]:.3f}]'
        )
        self.get_logger().info(
            f'双重融合 - 顶面中心: [{grasp_position[0]:.3f}, {grasp_position[1]:.3f}, {grasp_position[2]:.3f}]'
        )
        self.get_logger().info(
            f'误差估计: {error_info["total_error"]*1000:.1f}mm'
        )

        if not self._grasp_passes_stability_gate(grasp_position, final_normal, result_confidence):
            return

        if validate_position(grasp_position):
            self._send_position_with_tf(grasp_position, timestamp)
            self._last_sent_grasp = grasp_position.copy()
            self._last_sent_normal = final_normal.copy()
        else:
            self.get_logger().warn(f'位置超出范围: {format_position(grasp_position)}')
    
    def _handle_no_detection(self, timestamp):
        """处理无检测情况"""
        self._stable_grasp_counter = 0
        if self.use_kalman and self.tracker is not None:
            with self.tracker_lock:
                if not self.tracker.is_valid():
                    self.tracker = None
        
        # 发送无目标消息
        self.serial_comm.send_message('none\n')

    def _grasp_passes_stability_gate(
        self,
        grasp_position: np.ndarray,
        normal: np.ndarray,
        confidence: float,
    ) -> bool:
        """
        发送前稳定性闸门：
        1) 置信度低于阈值不发送；
        2) 与上一帧已发送结果相比，位移/法向突变过大不发送；
        3) 需要连续 N 帧满足条件后才发送。
        """
        if not self.enable_grasp_consistency_gate:
            return True

        if confidence < self.grasp_confidence_threshold:
            self._stable_grasp_counter = 0
            self.get_logger().warn(
                f'闸门阻止发送：置信度 {confidence:.2f} < {self.grasp_confidence_threshold:.2f}',
                throttle_duration_sec=0.5,
            )
            return False

        if self._last_sent_grasp is not None:
            jump = float(np.linalg.norm(grasp_position - self._last_sent_grasp))
            if jump > self.grasp_max_jump_m:
                self._stable_grasp_counter = 0
                self.get_logger().warn(
                    f'闸门阻止发送：位置跳变 {jump:.3f}m > {self.grasp_max_jump_m:.3f}m',
                    throttle_duration_sec=0.5,
                )
                return False

        if self._last_sent_normal is not None:
            n1 = normal / max(np.linalg.norm(normal), 1e-9)
            n2 = self._last_sent_normal / max(np.linalg.norm(self._last_sent_normal), 1e-9)
            dotp = float(np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0))
            angle = float(np.degrees(np.arccos(dotp)))
            if angle > self.grasp_max_normal_angle_deg:
                self._stable_grasp_counter = 0
                self.get_logger().warn(
                    f'闸门阻止发送：法向跳变 {angle:.1f}° > {self.grasp_max_normal_angle_deg:.1f}°',
                    throttle_duration_sec=0.5,
                )
                return False

        self._stable_grasp_counter += 1
        if self._stable_grasp_counter < self.grasp_min_confirm_frames:
            self.get_logger().info(
                f'闸门等待确认帧: {self._stable_grasp_counter}/{self.grasp_min_confirm_frames}',
                throttle_duration_sec=0.5,
            )
            return False

        return True
    
    def _send_position_with_tf(self, position: np.ndarray, timestamp):
        """TF转换到目标坐标系并发送位置"""
        point = PointStamped()
        point.header.frame_id = 'camera_link'
        point.header.stamp = timestamp
        point.point.x = float(position[0])
        point.point.y = float(position[1])
        point.point.z = float(position[2])
        
        try:
            transformed_point = self.tf_buffer.transform(
                point, self.target_output_frame, timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            final_pos = [
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            ]
            
            success = self.serial_comm.send_position(final_pos)
            
            # 日志控制
            self.log_counter += 1
            if self.log_counter % self.log_every_n_sends == 0:
                stats = self.performance_monitor.get_stats()
                self.get_logger().info(
                    f'发送位置({self.target_output_frame}): {format_position(np.array(final_pos))} | '
                    f'推理: {stats["avg_inference_ms"]:.1f}ms | '
                    f'回调处理: {stats["avg_processing_ms"]:.1f}ms | '
                    f'消息延迟: {stats["avg_callback_latency_ms"]:.1f}ms | '
                    f'跳帧: 1/{self.process_every_n_frames} | '
                    f'FPS: {stats["fps"]:.1f} | '
                    f'成功: {success}'
                )
        
        except Exception as e:
            self.get_logger().error(f'TF转换失败: {e}')
            if self.params['fallback_send_without_tf']:
                self.get_logger().warn('⚠️ TF不可用，降级发送 camera_link 坐标（单位：米）')
                self.serial_comm.send_position([float(position[0]), float(position[1]), float(position[2])])

    def _extract_camera_points_from_depth(self, depth_image: np.ndarray, bbox: list, position_3d: np.ndarray) -> Optional[np.ndarray]:
        """从深度图 ROI 区域提取相机坐标系下的 3D 点云
        
        Args:
            depth_image: 深度图 (H x W)
            bbox: [x_min, y_min, x_max, y_max] 检测框
            position_3d: 预估的 3D 位置（用于确定 ROI 深度范围）
            
        Returns:
            相机坐标系下的 3D 点云 (N x 3)，如果失败返回 None
        """
        try:
            x_min, y_min, x_max, y_max = map(int, bbox)
            h, w = depth_image.shape[:2]
            
            # 确保边界在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                return None
            
            # 获取 ROI 深度图
            roi_depth = depth_image[y_min:y_max, x_min:x_max]
            
            # 根据预估位置确定深度范围
            center_depth = position_3d[2]  # Z 坐标
            depth_range = 0.3  # ±30cm 范围
            min_depth = max(0.1, center_depth - depth_range)
            max_depth = center_depth + depth_range
            
            # 提取有效深度点
            valid_mask = (roi_depth > min_depth) & (roi_depth < max_depth) & np.isfinite(roi_depth)
            valid_y, valid_x = np.where(valid_mask)
            
            if len(valid_y) == 0:
                return None
            
            # 转换为图像坐标
            valid_x_global = valid_x + x_min
            valid_y_global = valid_y + y_min
            valid_depths = roi_depth[valid_y, valid_x]
            
            # 反投影到相机坐标系
            fx = self.camera_params['fx']
            fy = self.camera_params['fy']
            cx = self.camera_params['cx']
            cy = self.camera_params['cy']
            
            X = (valid_x_global - cx) * valid_depths / fx
            Y = (valid_y_global - cy) * valid_depths / fy
            Z = valid_depths
            
            camera_points = np.stack([X, Y, Z], axis=1).astype(np.float32)
            
            # 限制点数，避免过多
            max_points = 5000
            if len(camera_points) > max_points:
                indices = np.random.choice(len(camera_points), max_points, replace=False)
                camera_points = camera_points[indices]
            
            return camera_points
            
        except Exception as e:
            self.get_logger().warn(f'提取相机点云失败: {e}')
            return None

    def _prefilter_lidar_points(self, lidar_points: np.ndarray) -> np.ndarray:
        """雷达点云预过滤：范围/高度/横向裁剪，减少全量点干扰与算力开销。"""
        if not self.lidar_prefilter_enable or lidar_points is None or len(lidar_points) == 0:
            return lidar_points

        try:
            pts = lidar_points
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]
            r = np.sqrt(x * x + y * y + z * z)

            mask = (
                np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
                (r >= self.lidar_min_range_m) &
                (r <= self.lidar_max_range_m) &
                (np.abs(y) <= self.lidar_max_abs_y_m) &
                (np.abs(z) <= self.lidar_max_abs_z_m)
            )
            filtered = pts[mask]
            self.get_logger().info(
                f'雷达预过滤: {len(pts)} -> {len(filtered)} 点',
                throttle_duration_sec=1.0,
            )
            return filtered
        except Exception as e:
            self.get_logger().warn(f'雷达预过滤异常，回退原始点云: {e}')
            return lidar_points


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = R1VisionNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'节点运行异常: {e}')
    finally:
        # 清理资源
        if 'node' in locals():
            if hasattr(node, 'detector'):
                node.detector.cleanup()
            if hasattr(node, 'serial_comm'):
                node.serial_comm.close()
            
            # 性能统计
            if hasattr(node, 'performance_monitor'):
                stats = node.performance_monitor.get_stats()
                print(f'性能统计: 平均推理 {stats["avg_inference_ms"]:.1f}ms, '
                      f'平均FPS {stats["fps"]:.1f}')
            
            if hasattr(node, 'serial_comm'):
                comm_stats = node.serial_comm.get_stats()
                print(f'通信统计: 发送 {comm_stats["send_count"]} 次, '
                      f'错误率 {comm_stats["error_rate"]:.2%}')
            
            node.destroy_node()
        
        cleanup_gpu()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
