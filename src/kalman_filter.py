#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class KalmanFilter3D:
    def __init__(self):
        rospy.init_node('kalman_filter_3d', anonymous=True)

        # Subscribers for the three plane topics
        self.sub_xy = rospy.Subscriber('/xy_plane', Marker, self.callback_xy)
        self.sub_yz = rospy.Subscriber('/yz_plane', Marker, self.callback_yz)
        self.sub_xz = rospy.Subscriber('/xz_plane', Marker, self.callback_xz)

        # Publisher for the reconstructed 3D points
        self.pub_reconstructed = rospy.Publisher('/reconstructed_points', Marker, queue_size=10)

        # Storage for the latest data from each plane
        self.data_xy = []
        self.data_yz = []
        self.data_xz = []

        # Kalman filter parameters for each point
        self.filters = {}  # Dictionary to store filters for each point ID

    def initialize_filter(self, point_id, x, y, z):
        """Initialize a Kalman filter for a given point."""
        dt = 0.1  # Time step

        # State transition matrix
        A = np.eye(6)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        # Measurement matrix
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        # Covariance matrices
        Q = np.eye(6) * 0.001  # Process noise covariance
        R = np.eye(3) * 0.01   # Measurement noise covariance
        P = np.eye(6)          # State covariance

        # Initial state
        x_state = np.zeros((6, 1))
        x_state[0, 0] = x
        x_state[1, 0] = y
        x_state[2, 0] = z

        return {
            "A": A,
            "H": H,
            "Q": Q,
            "R": R,
            "P": P,
            "x": x_state
        }

    def callback_xy(self, msg):
        self.data_xy = [(i, p.x, p.y) for i, p in enumerate(msg.points)]
        self.process_data()

    def callback_yz(self, msg):
        self.data_yz = [(i, p.y, p.z) for i, p in enumerate(msg.points)]
        self.process_data()

    def callback_xz(self, msg):
        self.data_xz = [(i, p.x, p.z) for i, p in enumerate(msg.points)]
        self.process_data()

    def process_data(self):
        if not (self.data_xy and self.data_yz and self.data_xz):
            return  # Wait until data from all planes is available

        # Reconstruct 3D points
        reconstructed_points = []
        point_ids = set(i for i, _, _ in self.data_xy)

        for point_id in point_ids:
            # Extract measurements for this point ID
            x, y = next((x, y) for i, x, y in self.data_xy if i == point_id)
            _, z = next((x, z) for i, x, z in self.data_xz if i == point_id)

            # Initialize filter if it doesn't exist
            if point_id not in self.filters:
                self.filters[point_id] = self.initialize_filter(point_id, x, y, z)

            # Retrieve the filter for this point
            kf = self.filters[point_id]
            z_k = np.array([[x], [y], [z]])  # Measurement vector

            # Update the Kalman filter
            y_k = z_k - np.dot(kf["H"], kf["x"])  # Innovation
            S = np.dot(kf["H"], np.dot(kf["P"], kf["H"].T)) + kf["R"]  # Innovation covariance
            K = np.dot(kf["P"], np.dot(kf["H"].T, np.linalg.inv(S)))  # Kalman gain

            kf["x"] += np.dot(K, y_k)  # Update state
            kf["P"] = np.dot(np.eye(6) - np.dot(K, kf["H"]), kf["P"])  # Update covariance

            # Predict the next state
            kf["x"] = np.dot(kf["A"], kf["x"])
            kf["P"] = np.dot(kf["A"], np.dot(kf["P"], kf["A"].T)) + kf["Q"]

            # Add the estimated position to the list
            reconstructed_points.append(Point(x=kf["x"][0, 0], y=kf["x"][1, 0], z=kf["x"][2, 0]))

        # Publish the reconstructed points
        self.publish_reconstructed_points(reconstructed_points)

    def publish_reconstructed_points(self, points):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "reconstructed_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.2  # Point size
        marker.scale.y = 0.2
        marker.color.a = 1.0  # Alpha (transparency)
        marker.color.r = 0.0  # Red color
        marker.color.g = 0.0  # Green color
        marker.color.b = 1.0  # Blue color

        marker.points = points
        self.pub_reconstructed.publish(marker)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        kalman_filter = KalmanFilter3D()
        kalman_filter.run()
    except rospy.ROSInterruptException:
        pass
