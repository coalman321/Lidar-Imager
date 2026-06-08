from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ouster_params_file = PathJoinSubstitution([
        FindPackageShare('lidar_imager'),
        'config',
        'driver_params.yaml'
    ])

    ouster_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ouster_ros'),
                'launch',
                'driver.launch.py'
            ])
        ),
        launch_arguments={
            'params_file': ouster_params_file
        }.items()
    )

    lidar_imager_node = Node(
        package='lidar_imager',
        executable='lidar_imager',
        name='lidar_imager',
        output='screen'
    )

    return LaunchDescription([
        ouster_launch,
        lidar_imager_node,
    ])
