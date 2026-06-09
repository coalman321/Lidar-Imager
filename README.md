# Lidar-Imager
ROS2 lidar projection imager

![UI Image of the tool](./doc/ui.png)

## What is this tool?
This tool is designed to project 3D LiDAR point clouds onto a 2D image plane, creating a visual representation of the LiDAR data. It subscribes to a ROS2 topic that provides PointCloud2 messages, processes the point cloud data, and generates a PNG image.

The PNG Image is then exported as a PNG with a background image, or as a PDF with two circular crops. 
The circular crops are sized for creating a printed button, and the exported png can be printed on photo paper.

This tool was built as a fun demo of how robots understand the world around them, and how we can visualize that understanding. It is not intended for any specific use case, but rather as a demonstration of the capabilities of ROS2 and LiDAR data processing.

## Configuration

The tool is configured via the configure button in the UI. The options include:
- min and max Z for color mapping
- point size for the projected points
- background image for the PNG export
- location of the png export on the background image
- output directory for the exported images
- text font and color for the names in the pdf and png exports

![Configuration UI](./doc/config.png)

## Exported formats
The tool can export the projected point cloud in two formats:
- PNG: A single image with the projected point cloud overlaid on a background image. Also includes a name for the point cloud, which can be customized.
[PNG Export Example](./doc/png_export.png)
- PDF: A single-page PDF with two circular crops of the projected point cloud, designed for printing. also includes a name for the point cloud, which can be customized.
[PDF Export Example](./doc/pdf_export.pdf)


## Setup
This package was designed for ROS2 Jazzy on Ubuntu 24.04. It should work with other versions of ROS2 but may need some modifications. Clone this package into your ROS2 workspace and clone the `ouster-ros` package if you want to run this package with an Ouster lidar sensor.

```
cd ros2_ws/src
vcs import < Lidar-Imager/lidar_imager/setup/lidar_imager.repos --recursive
```

Compile
```
cd ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Source the workspace
```
source install/setup.bash
```

## Operation

### Lidar_Imager Node Only
The node defaults to a pointcloud2 topic of `/ouster/points`.
```
ros2 run lidar_imager lidar_imager
```

### Ouster Example
Ensure to properly configure the parameters for the ouster. This launch file will utilize the configuration in this repo at `Lidar-Imager/lidar_imager/config/driver_params.yaml`, not the configuration located within `ouster-ros`. 

Start the launch file that will start both the ouster lidar driver and the lidar_imager.

```
ros2 launch lidar_imager lidar_imager.launch.py
```