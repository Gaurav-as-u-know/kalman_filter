#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import random

class MultiPointPublisher:
    def __init__(self):
        rospy.init_node('multi_point_publisher', anonymous=True)

        # Publisher for true positions
        self.pub_marker = rospy.Publisher('/true_positions', Marker, queue_size=10)

        # Parameters for the number of points, space size, and motion
        self.num_points = rospy.get_param('~num_points', 10)  # Default to 10 points
        self.space_size = rospy.get_param('~space_size', 5)  # Cube size (e.g., -5 to 5)
        self.points = self.initialize_points()
        self.velocities = self.initialize_velocities()
        self.update_rate = rospy.get_param('~update_rate', 23)  # Update rate in Hz

    def initialize_points(self):
        """Initialize points with random positions within a cubic space."""
        return [
            {
                'x': random.uniform(-self.space_size, self.space_size),
                'y': random.uniform(-self.space_size, self.space_size),
                'z': random.uniform(-self.space_size, self.space_size)
            }
            for _ in range(self.num_points)
        ]

    def initialize_velocities(self):
        """Assign slow random velocities to points."""
        speed = float(0.06)
        return [
            {
                
                'vx': random.uniform(-speed, speed),
                'vy': random.uniform(-speed, speed),
                'vz': random.uniform(-speed, speed)
            }
            for _ in range(self.num_points)
        ]

    def update_positions(self):
        """Update point positions based on their velocities and bounce at boundaries."""
        for i, point in enumerate(self.points):
            # Update positions
            point['x'] += self.velocities[i]['vx']
            point['y'] += self.velocities[i]['vy']
            point['z'] += self.velocities[i]['vz']

            # Bounce the points if they hit the boundaries of the cube
            if point['x'] > self.space_size or point['x'] < -self.space_size:
                self.velocities[i]['vx'] *= -1  # Reverse velocity in x-direction

            if point['y'] > self.space_size or point['y'] < -self.space_size:
                self.velocities[i]['vy'] *= -1  # Reverse velocity in y-direction

            if point['z'] > self.space_size or point['z'] < -self.space_size:
                self.velocities[i]['vz'] *= -1  # Reverse velocity in z-direction

    def publish_marker(self):
        """Publish all points as a single Marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = 0  # You can change this if you want unique IDs for each group of points
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Marker properties
        marker.scale.x = 0.2  # Point size
        marker.scale.y = 0.2
        marker.color.a = 1.0  # Alpha (transparency)
        marker.color.r = 1.0  # Red color
        marker.color.g = 1.0  # Green color
        marker.color.b = 0.0  # Yellow (Red + Green)

        # Add all points to the marker
        for i, point in enumerate(self.points):
            p = Point()
            p.x = point['x']
            p.y = point['y']
            p.z = point['z']
            marker.points.append(p)

        self.pub_marker.publish(marker)

    def run(self):
        """Main loop to update and publish points."""
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self.update_positions()
            self.publish_marker()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = MultiPointPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
