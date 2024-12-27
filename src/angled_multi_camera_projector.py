#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import numpy as np

class PointRotationPublisher:
    def __init__(self):
        rospy.init_node('point_rotation_publisher', anonymous=True)

        # Publisher for the rotated point
        self.pub_rotated_point = rospy.Publisher('/rotated_point', Marker, queue_size=10)

        # Parameters
        self.update_rate = rospy.get_param('~update_rate', 30)  # Update rate in Hz
        self.angle_x = rospy.get_param('~angle_x', 30)  # Rotation around x-axis in degrees
        self.angle_y = rospy.get_param('~angle_y', 45)  # Rotation around y-axis in degrees
        self.angle_z = rospy.get_param('~angle_z', 60)  # Rotation around z-axis in degrees

        # Original point
        self.original_point = Point(2.0, 2.0, 2.0)

    def rotate_point(self, x, y, z, angle_x, angle_y, angle_z):
        """Rotate a point by given angles (in degrees) around x, y, z axes."""
        # Convert angles to radians
        angle_x = math.radians(angle_x)
        angle_y = math.radians(angle_y)
        angle_z = math.radians(angle_z)

        # Rotation matrices for X, Y, Z axes
        R_x = [
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x), math.cos(angle_x)]
        ]
        R_y = [
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]
        ]
        R_z = [
            [math.cos(angle_z), -math.sin(angle_z), 0],
            [math.sin(angle_z), math.cos(angle_z), 0],
            [0, 0, 1]
        ]

        # Combined rotation matrix
        R = np.dot(R_z, np.dot(R_y, R_x))

        # Apply the rotation
        rotated = np.dot(R, [x, y, z])
        return rotated[0], rotated[1], rotated[2]

    def publish_rotated_point(self):
        """Publish the rotated point as a Marker."""
        # Rotate the original point
        rotated_x, rotated_y, rotated_z = self.rotate_point(self.original_point.x, self.original_point.y, self.original_point.z, self.angle_x, self.angle_y, self.angle_z)

        # Create the marker for the rotated point
        marker = Marker()
        marker.header.frame_id = "map"  # Ensure this matches the Fixed Frame in RViz
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rotated_point"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Marker properties
        marker.scale.x = 0.2  # Point size
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # Alpha (transparency)
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0  # Green color
        marker.color.b = 0.0  # Blue color

        # Add the rotated point to the marker
        rotated_point = Point(rotated_x, rotated_y, rotated_z)
        marker.points.append(rotated_point)

        # Publish the marker
        self.pub_rotated_point.publish(marker)

    def run(self):
        """Main loop to update and publish the rotated point."""
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self.publish_rotated_point()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = PointRotationPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
