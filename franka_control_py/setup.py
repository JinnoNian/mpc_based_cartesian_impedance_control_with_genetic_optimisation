from setuptools import setup
import os
from glob import glob

package_name = 'franka_control_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'franka_msgs',
        'geometry_msgs',
        'rosidl_default_generators',
        'cvxpy',
        'scipy',        # 添加 scipy 依赖
        'deap',         # 添加 deap 依赖
        'pinocchio'     # 添加 pinocchio 依赖
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Franka control with Cartesian Impedance Controller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'franka_control = franka_control_py.franka_control:main',
        ],
    },
)

