#!/usr/bin/env python3

"""
R1 Vision集成FAST-LIO启动文件
mid360: livox_ros_driver2 + fast_lio + r1_vision
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # 获取包路径
    r1_vision_dir = get_package_share_directory('r1_vision')
    fast_lio_dir = get_package_share_directory('fast_lio')

    # 配置文件路径（mid360）
    fast_lio_config = os.path.join(fast_lio_dir, 'config', 'mid360.yaml')

    # 声明启动参数
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(r1_vision_dir, 'models', 'best.pt'),
        description='YOLO模型文件路径'
    )

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='串口设备路径'
    )

    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='9600',
        description='串口波特率'
    )

    use_gpu_arg = DeclareLaunchArgument(
        'use_gpu',
        default_value='true',
        description='是否使用GPU加速'
    )

    mock_serial_arg = DeclareLaunchArgument(
        'mock_serial',
        default_value='false',
        description='是否使用Mock串口（测试模式）'
    )

    use_kalman_arg = DeclareLaunchArgument(
        'use_kalman',
        default_value='true',
        description='是否启用Kalman滤波跟踪'
    )

    enable_light_adaptation_arg = DeclareLaunchArgument(
        'enable_light_adaptation',
        default_value='true',
        description='是否启用光照自适应'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='是否使用仿真时间'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='是否启动RViz可视化'
    )
    depth_window_arg = DeclareLaunchArgument(
        'depth_window',
        default_value='5',
        description='深度滤波窗口大小'
    )
    normal_window_arg = DeclareLaunchArgument(
        'normal_window',
        default_value='7',
        description='法向量估计窗口大小'
    )
    normal_window_size_arg = DeclareLaunchArgument(
        'normal_window_size',
        default_value='5',
        description='法向量平滑窗口大小'
    )
    r1_config_file_arg = DeclareLaunchArgument(
        'r1_config_file',
        default_value=os.path.join(r1_vision_dir, 'config', 'tf_config.yaml'),
        description='R1集成配置文件（含 lidar_to_camera 外参）'
    )

    # 话题名参数（默认mid360标准话题），避免相机内参超时
    topic_rgb_arg = DeclareLaunchArgument(
        'topic_rgb',
        default_value='/camera/camera/color/image_raw',
        description='RGB图像话题'
    )
    topic_depth_arg = DeclareLaunchArgument(
        'topic_depth',
        default_value='/camera/camera/aligned_depth_to_color/image_raw',
        description='深度图像话题（对齐后）'
    )
    topic_lidar_points_arg = DeclareLaunchArgument(
        'topic_lidar_points',
        default_value='/livox/lidar',
        description='雷达点云话题（mid360）'
    )
    topic_imu_arg = DeclareLaunchArgument(
        'topic_imu',
        default_value='/livox/imu',
        description='IMU话题（mid360）'
    )
    topic_odom_arg = DeclareLaunchArgument(
        'topic_odom',
        default_value='/Odometry',
        description='里程计话题'
    )
    sync_slop_arg = DeclareLaunchArgument(
        'sync_slop',
        default_value='0.05',
        description='时间同步容差（秒）'
    )
    body_to_base_x_arg = DeclareLaunchArgument('body_to_base_x', default_value='0.0', description='body->base_link 平移x')
    body_to_base_y_arg = DeclareLaunchArgument('body_to_base_y', default_value='0.0', description='body->base_link 平移y')
    body_to_base_z_arg = DeclareLaunchArgument('body_to_base_z', default_value='0.0', description='body->base_link 平移z')
    body_to_base_roll_arg = DeclareLaunchArgument('body_to_base_roll', default_value='0', description='body->base_link 旋转roll')
    body_to_base_pitch_arg = DeclareLaunchArgument('body_to_base_pitch', default_value='0', description='body->base_link 旋转pitch')
    body_to_base_yaw_arg = DeclareLaunchArgument('body_to_base_yaw', default_value='0', description='body->base_link 旋转yaw')

    camera_to_base_x_arg = DeclareLaunchArgument('camera_to_base_x', default_value='0.0', description='camera_link->base_link 平移x')
    camera_to_base_y_arg = DeclareLaunchArgument('camera_to_base_y', default_value='0.0', description='camera_link->base_link 平移y')
    camera_to_base_z_arg = DeclareLaunchArgument('camera_to_base_z', default_value='0.2', description='camera_link->base_link 平移z')
    camera_to_base_roll_arg = DeclareLaunchArgument('camera_to_base_roll', default_value='0', description='camera_link->base_link 旋转roll')
    camera_to_base_pitch_arg = DeclareLaunchArgument('camera_to_base_pitch', default_value='0', description='camera_link->base_link 旋转pitch')
    camera_to_base_yaw_arg = DeclareLaunchArgument('camera_to_base_yaw', default_value='0', description='camera_link->base_link 旋转yaw')

    lidar_msg_type_arg = DeclareLaunchArgument(
        'lidar_msg_type',
        default_value='custom',
        description='雷达消息类型 (custom 或 pointcloud2)'
    )

    # mid360 驱动由 livox_ros_driver2 提供，通常单独启动或已在运行
    # 若需要在此一并启动，取消下面注释（需要已安装 livox_ros_driver2）：
    # livox_driver = Node(
    #     package='livox_ros_driver2',
    #     executable='livox_ros_driver2_node',
    #     name='livox_ros_driver2_node',
    #     output='screen',
    # )

    # 静态TF发布器：FAST-LIO 的 body frame 到 base_link
    # FAST-LIO 发布 odom: camera_init -> body；body 即 IMU/LiDAR body frame
    # 这里将 body 与 base_link 对齐（假设同一坐标系），使 r1_vision 的 TF 查询可用
    body_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_base_tf',
        arguments=[
            LaunchConfiguration('body_to_base_x'),
            LaunchConfiguration('body_to_base_y'),
            LaunchConfiguration('body_to_base_z'),
            LaunchConfiguration('body_to_base_roll'),
            LaunchConfiguration('body_to_base_pitch'),
            LaunchConfiguration('body_to_base_yaw'),
            'body',
            'base_link',
        ],
        output='screen'
    )

    # 相机静态TF发布器：camera_link 到 base_link
    # r1_vision.update_gravity_vector() 会 lookup_transform('camera_link', 'base_link')
    # ⚠️ 请根据实际安装位置修改 translation/rotation 参数
    camera_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
            LaunchConfiguration('camera_to_base_x'),
            LaunchConfiguration('camera_to_base_y'),
            LaunchConfiguration('camera_to_base_z'),
            LaunchConfiguration('camera_to_base_roll'),
            LaunchConfiguration('camera_to_base_pitch'),
            LaunchConfiguration('camera_to_base_yaw'),
            'camera_link',
            'base_link',
        ],
        output='screen'
    )

    # FAST-LIO节点：使用其原生的 mapping.launch.py，以确保完全兼容它的参数体系
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('fast_lio'),
                'launch',
                'mapping.launch.py'
            ])
        ),
        launch_arguments={
            'config_file': 'mid360.yaml',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'rviz': 'false' # R1 Vision 自己已经启动了 RViz，关闭 fast_lio 的避免双开
        }.items()
    )

    # RealSense 相机启动已移除（使用硬编码内参）

    # R1 Vision节点
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
            'log_every_n_sends': 10,
            'log_level': 'INFO',

            # 明确传入 topic，避免“接上了但 r1_vision 没收到”的情况
            'topic_rgb': LaunchConfiguration('topic_rgb'),
            'topic_depth': LaunchConfiguration('topic_depth'),
            'topic_lidar_points': LaunchConfiguration('topic_lidar_points'),
            'topic_imu': LaunchConfiguration('topic_imu'),
            'topic_odom': LaunchConfiguration('topic_odom'),
            'sync_slop': LaunchConfiguration('sync_slop'),
            'lidar_msg_type': LaunchConfiguration('lidar_msg_type'),
            'depth_window': LaunchConfiguration('depth_window'),
            'normal_window': LaunchConfiguration('normal_window'),
            'normal_window_size': LaunchConfiguration('normal_window_size'),
            'r1_config_file': LaunchConfiguration('r1_config_file'),
            'fallback_send_without_tf': True,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # RViz可视化节点（可选）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(r1_vision_dir, 'rviz', 'r1_vision.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen'
    )

    return LaunchDescription([
        # 参数声明
        model_path_arg,
        serial_port_arg,
        baud_rate_arg,
        use_gpu_arg,
        mock_serial_arg,
        use_kalman_arg,
        enable_light_adaptation_arg,
        use_sim_time_arg,
        rviz_arg,
        depth_window_arg,
        normal_window_arg,
        normal_window_size_arg,
        r1_config_file_arg,
        topic_rgb_arg,
        topic_depth_arg,
        topic_lidar_points_arg,
        topic_imu_arg,
        topic_odom_arg,
        sync_slop_arg,
        body_to_base_x_arg,
        body_to_base_y_arg,
        body_to_base_z_arg,
        body_to_base_roll_arg,
        body_to_base_pitch_arg,
        body_to_base_yaw_arg,
        camera_to_base_x_arg,
        camera_to_base_y_arg,
        camera_to_base_z_arg,
        camera_to_base_roll_arg,
        camera_to_base_pitch_arg,
        camera_to_base_yaw_arg,
        lidar_msg_type_arg,

        # 节点（mid360驱动由 livox_ros_driver2 单独管理，不在此启动）
        body_to_base_tf,
        camera_to_base_tf,
        fast_lio_launch,
        r1_vision_node,
        rviz_node
    ])
