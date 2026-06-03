"""
main.py — Entry point for the LiDAR Imager application.

Starts the ROS2 executor in a background daemon thread, then launches the
tkinter GUI on the main thread.  On window close both threads are shut down
cleanly.
"""

from __future__ import annotations

import sys
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .app import LidarImagerApp
from .ros_node import PointCloudNode


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PointCloudNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # Run ROS2 executor in a daemon background thread so it dies automatically
    # if the main thread (tkinter) exits unexpectedly.
    stop_event = threading.Event()

    def spin_thread() -> None:
        try:
            while not stop_event.is_set():
                executor.spin_once(timeout_sec=0.05)
        except Exception:  # noqa: BLE001
            pass
        finally:
            executor.remove_node(node)

    ros_thread = threading.Thread(target=spin_thread, daemon=True)
    ros_thread.start()

    try:
        app = LidarImagerApp(node=node)
        app.mainloop()
    finally:
        stop_event.set()
        ros_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
