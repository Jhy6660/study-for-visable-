#!/usr/bin/env python3

"""
R1 Vision启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('r1_vision')

    args = [
        DeclareLaunchArgument('model_path',
            default_value=os.path.join(package_dir, 'models', 'best.pt'),
            description='YOLO模型文件路径'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0',
            description='串口设备路径'),
        DeclareLaunchArgument('baud_rate', default_value='9600',
            description='串口波特率'),
        DeclareLaunchArgument('use_gpu', default_value='true',
            description='是否使用GPU加速'),
        DeclareLaunchArgument('mock_serial', default_value='false',
            description='是否使用Mock串口（测试模式）'),
        DeclareLaunchArgument('use_kalman', default_value='true',
            description='是否启用Kalman滤波跟踪'),
        DeclareLaunchArgument('enable_light_adaptation', default_value='true',
            description='是否启用光照自适应'),
        # 话题名参数（默认mid360标准话题）
        DeclareLaunchArgument('topic_rgb', default_value='/camera/camera/color/image_raw',
            description='RGB图像话题'),
        DeclareLaunchArgument('topic_depth', default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='深度图像话题（对齐后）'),
        DeclareLaunchArgument('topic_lidar_points', default_value='/livox/lidar',
            description='雷达点云话题（mid360）'),
        DeclareLaunchArgument('topic_imu', default_value='/livox/imu',
            description='IMU话题（mid360）'),
        DeclareLaunchArgument('topic_odom', default_value='/Odometry',
            description='里程计话题'),
        DeclareLaunchArgument('topic_camera_info', default_value='/camera/camera/color/camera_info',
            description='相机内参话题'),
        DeclareLaunchArgument('sync_slop', default_value='0.03',
            description='时间同步容差（秒）'),
        DeclareLaunchArgument('lidar_msg_type', default_value='custom',
            description='雷达消息类型 (custom 或 pointcloud2)'),
        DeclareLaunchArgument('r1_config_file', 
            default_value=os.path.join(package_dir, 'config', 'tf_config.yaml'),
            description='R1集成配置文件（含 lidar_to_camera 外参）'),
        DeclareLaunchArgument('depth_window', default_value='5',
            description='深度滤波窗口大小'),
        DeclareLaunchArgument('normal_window', default_value='7',
            description='法向量估计窗口大小'),
    ]

    # RealSense 相机启动已移除（使用硬编码内参）

    r1_vision_node = Node(
        package='r1_vision',
        executable='r1_vision_node',
        name='r1_vision_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'use_gpu': LaunchConfiguration('use_gpu'),
            'mock_serial': LaunchConfiguration('mock_serial'),
            'use_kalman': LaunchConfiguration('use_kalman'),
            'enable_light_adaptation': LaunchConfiguration('enable_light_adaptation'),
            'topic_rgb': LaunchConfiguration('topic_rgb'),
            'topic_depth': LaunchConfiguration('topic_depth'),
            'topic_lidar_points': LaunchConfiguration('topic_lidar_points'),
            'topic_imu': LaunchConfiguration('topic_imu'),
            'topic_odom': LaunchConfiguration('topic_odom'),
            'sync_slop': LaunchConfiguration('sync_slop'),
            'lidar_msg_type': LaunchConfiguration('lidar_msg_type'),
            'r1_config_file': LaunchConfiguration('r1_config_file'),
            'depth_window': LaunchConfiguration('depth_window'),
            'normal_window': LaunchConfiguration('normal_window'),
            'variance_threshold': 100.0,
            'process_every_n_frames': 2,
            'log_every_n_sends': 10,
            'log_level': 'INFO',
            'fallback_send_without_tf': True,
        }]
    )

    return LaunchDescription(args + [r1_vision_node])
