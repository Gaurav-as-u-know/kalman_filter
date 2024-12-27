#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import random

class MultiPointPublisher:
    def __init__(self):
        rospy.init_node('multi_point_publisher', anonymous=True)

        # Publishers for the true positions and projected positions on the planes
        self.pub_marker = rospy.Publisher('/true_positions', Marker, queue_size=10)
        self.pub_xy_plane = rospy.Publisher('/xy_plane', Marker, queue_size=10)
        self.pub_yz_plane = rospy.Publisher('/yz_plane', Marker, queue_size=10)
        self.pub_xz_plane = rospy.Publisher('/xz_plane', Marker, queue_size=10)

        # Parameters for the number of points, space size, and motion
        self.num_points = rospy.get_param('~num_points', 20)  # Default to 10 points
        self.space_size = rospy.get_param('~space_size', 5)  # Cube size (e.g., -5 to 5)
        self.points = self.initialize_points()
        self.velocities = self.initialize_velocities()
        self.update_rate = rospy.get_param('~update_rate', 30)  # Update rate in Hz

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
        speed = 0.05
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

    def add_noise(self, point, noise_level=0.1):
        """Add random noise to a point."""
        noise = random.uniform(-noise_level, noise_level)
        point.x += noise
        point.y += noise
        point.z += noise
        return point

    def project_points(self):
        """Project points onto the three orthogonal planes."""
        xy_points = []
        yz_points = []
        xz_points = []
        
        for point in self.points:
            # Project to X-Y plane (set z = 0)
            xy_point = Point()
            xy_point.x = point['x']
            xy_point.y = point['y']
            xy_point.z = 0
            xy_points.append(self.add_noise(xy_point))  # Add noise

            # Project to Y-Z plane (set x = 0)
            yz_point = Point()
            yz_point.x = 0
            yz_point.y = point['y']
            yz_point.z = point['z']
            yz_points.append(self.add_noise(yz_point))  # Add noise

            # Project to X-Z plane (set y = 0)
            xz_point = Point()
            xz_point.x = point['x']
            xz_point.y = 0
            xz_point.z = point['z']
            xz_points.append(self.add_noise(xz_point))  # Add noise

        return xy_points, yz_points, xz_points

    def publish_marker(self):
        """Publish all points as a single Marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Marker properties
        marker.scale.x = 0.2  # Point size
        marker.scale.y = 0.2
        marker.color.a = 1.0  # Alpha (transparency)
        marker.color.r = 1.0  # Red color
        marker.color.g = 1.0 # Green color
        marker.color.b = 1.0  # Yellow (Red + Green)

        # Add all points to the marker
        for point in self.points:
            p = Point()
            p.x = point['x']
            p.y = point['y']
            p.z = point['z']
            marker.points.append(p)

        self.pub_marker.publish(marker)

    def publish_projected_points(self, xy_points, yz_points, xz_points):
        """Publish the projected points onto the three planes."""
        # Publish X-Y Plane
        marker_xy = Marker()
        marker_xy.header.frame_id = "map"
        marker_xy.header.stamp = rospy.Time.now()
        marker_xy.ns = "xy_plane"
        marker_xy.id = 1
        marker_xy.type = Marker.POINTS
        marker_xy.action = Marker.ADD
        marker_xy.scale.x = 0.1
        marker_xy.scale.y = 0.1
        marker_xy.color.a = 1.0
        marker_xy.color.r = 0.0
        marker_xy.color.g = 1.0
        marker_xy.color.b = 0.0

        for point in xy_points:
            marker_xy.points.append(point)
        self.pub_xy_plane.publish(marker_xy)

        # Publish Y-Z Plane
        marker_yz = Marker()
        marker_yz.header.frame_id = "map"
        marker_yz.header.stamp = rospy.Time.now()
        marker_yz.ns = "yz_plane"
        marker_yz.id = 2
        marker_yz.type = Marker.POINTS
        marker_yz.action = Marker.ADD
        marker_yz.scale.x = 0.1
        marker_yz.scale.y = 0.1
        marker_yz.color.a = 1.0
        marker_yz.color.r = 0.0
        marker_yz.color.g = 0.0
        marker_yz.color.b = 1.0

        for point in yz_points:
            marker_yz.points.append(point)
        self.pub_yz_plane.publish(marker_yz)

        # Publish X-Z Plane
        marker_xz = Marker()
        marker_xz.header.frame_id = "map"
        marker_xz.header.stamp = rospy.Time.now()
        marker_xz.ns = "xz_plane"
        marker_xz.id = 3
        marker_xz.type = Marker.POINTS
        marker_xz.action = Marker.ADD
        marker_xz.scale.x = 0.1
        marker_xz.scale.y = 0.1
        marker_xz.color.a = 1.0
        marker_xz.color.r = 1.0
        marker_xz.color.g = 0.0
        marker_xz.color.b = 1.0

        for point in xz_points:
            marker_xz.points.append(point)
        self.pub_xz_plane.publish(marker_xz)

    def run(self):
        """Main loop to update and publish points."""
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self.update_positions()
            self.publish_marker()
            xy_points, yz_points, xz_points = self.project_points()
            self.publish_projected_points(xy_points, yz_points, xz_points)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = MultiPointPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
