from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'r1_vision'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'PyYAML>=6.0',
        'pyserial>=3.5',
        'scipy>=1.7.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'ultralytics>=8.0.0',
    ],
    zip_safe=True,
    maintainer='R1 Developer',
    maintainer_email='developer@example.com',
    description='R1模块化视觉检测系统 - 基于YOLO和深度相机的目标检测与跟踪',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'r1_vision_node = r1_vision.r1_vision_node:main',
        ],
    },
)
