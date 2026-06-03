"""
ros_node.py — ROS2 subscriber node for PointCloud2 messages.

Runs inside a background thread via rclpy.spin().
Thread-safe access to the latest point cloud via get_latest_points().
"""

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

POINTCLOUD_TOPIC = '/demo_pointcloud'


class PointCloudNode(Node):
    """Subscribes to a PointCloud2 topic and stores the latest points."""

    def __init__(self):
        super().__init__('lidar_imager_node')
        self._lock = threading.Lock()
        self._latest_points: np.ndarray | None = None
        self._frame_count: int = 0
        self._last_stamp: float = 0.0

        self._sub = self.create_subscription(
            PointCloud2,
            POINTCLOUD_TOPIC,
            self._callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.get_logger().info(f'Subscribed to {POINTCLOUD_TOPIC}')

    # ------------------------------------------------------------------
    # Subscriber callback (background thread)
    # ------------------------------------------------------------------

    def _callback(self, msg: PointCloud2) -> None:
        """Parse incoming PointCloud2 and cache XYZ as (N, 3) float32."""
        try:
            # read_points_numpy returns a plain (N, 3) unstructured float array
            # via structured_to_unstructured — field_names order is x, y, z
            xyz = pc2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            ).astype(np.float32)

            if xyz.size == 0 or xyz.ndim != 2 or xyz.shape[1] < 3:
                return

            stamp = (
                msg.header.stamp.sec
                + msg.header.stamp.nanosec * 1e-9
            )

            with self._lock:
                self._latest_points = xyz
                self._frame_count += 1
                self._last_stamp = stamp

        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'PointCloud parse error: {exc}')

    # ------------------------------------------------------------------
    # Public accessors (called from main thread)
    # ------------------------------------------------------------------

    def get_latest_points(self) -> np.ndarray | None:
        """Return a copy of the latest XYZ array, or None if none received."""
        with self._lock:
            if self._latest_points is None:
                return None
            return self._latest_points.copy()

    def get_status(self) -> dict:
        """Return lightweight status info for the status bar."""
        with self._lock:
            return {
                'topic': POINTCLOUD_TOPIC,
                'frame_count': self._frame_count,
                'point_count': (
                    len(self._latest_points)
                    if self._latest_points is not None
                    else 0
                ),
                'last_stamp': self._last_stamp,
            }
