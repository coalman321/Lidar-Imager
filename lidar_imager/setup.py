from setuptools import setup

package_name = 'lidar_imager'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='coalman321',
    maintainer_email='coalman321@todo.todo',
    description='ROS2 LiDAR front-view projection imager with tkinter GUI',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_imager = lidar_imager.main:main',
        ],
    },
)
