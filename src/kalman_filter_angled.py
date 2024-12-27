#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class KalmanFilter3D:
    def __init__(self):
        rospy.init_node('kalman_filter_3d', anonymous=True)

        # Subscribers for the three rotated plane topics
        self.sub_plane_1 = rospy.Subscriber('/plane_1', Marker, self.callback_plane_1)
        self.sub_plane_2 = rospy.Subscriber('/plane_2', Marker, self.callback_plane_2)
        self.sub_plane_3 = rospy.Subscriber('/plane_3', Marker, self.callback_plane_3)

        # Publisher for the reconstructed 3D points
        self.pub_reconstructed = rospy.Publisher('/reconstructed_points', Marker, queue_size=10)

        # Storage for the latest data from each plane
        self.data_plane_1 = []
        self.data_plane_2 = []
        self.data_plane_3 = []

        # Rotation matrices for the planes (angles must match the publisher)
        self.rotation_matrices = self.calculate_rotation_matrices()

        # Kalman filter parameters for each point
        self.filters = {}  # Dictionary to store filters for each point ID

    def calculate_rotation_matrices(self):
        """Pre-calculate rotation matrices for the planes."""
        angles = rospy.get_param('~angles', [30, 45, 60])  # Angles for the planes in degrees
        matrices = []
        for angle in angles:
            radians = np.radians(angle)
            matrix = np.array([
                [np.cos(radians), -np.sin(radians), 0],
                [np.sin(radians), np.cos(radians),  0],
                [0,               0,                1]
            ])
            matrices.append(matrix)
        return matrices

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

    def callback_plane_1(self, msg):
        self.data_plane_1 = [(i, p.x, p.y) for i, p in enumerate(msg.points)]
        self.process_data()

    def callback_plane_2(self, msg):
        self.data_plane_2 = [(i, p.x, p.y) for i, p in enumerate(msg.points)]
        self.process_data()

    def callback_plane_3(self, msg):
        self.data_plane_3 = [(i, p.x, p.y) for i, p in enumerate(msg.points)]
        self.process_data()

    def process_data(self):
        if not (self.data_plane_1 and self.data_plane_2 and self.data_plane_3):
            return  # Wait until data from all planes is available

        # Reconstruct 3D points
        reconstructed_points = []
        point_ids = set(i for i, _, _ in self.data_plane_1)

        for point_id in point_ids:
            # Extract measurements for this point ID
            p1 = next((x, y) for i, x, y in self.data_plane_1 if i == point_id)
            p2 = next((x, y) for i, x, y in self.data_plane_2 if i == point_id)
            p3 = next((x, y) for i, x, y in self.data_plane_3 if i == point_id)

            # Combine measurements into 3D using rotation matrices
            point_3d = self.triangulate_points(p1, p2, p3)

            # Initialize filter if it doesn't exist
            if point_id not in self.filters:
                self.filters[point_id] = self.initialize_filter(point_id, *point_3d)

            # Retrieve the filter for this point
            kf = self.filters[point_id]
            z_k = np.array([[point_3d[0]], [point_3d[1]], [point_3d[2]]])  # Measurement vector

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

    def triangulate_points(self, p1, p2, p3):
        """Combine points from rotated planes into a single 3D point."""
        p1_rotated = np.dot(np.linalg.inv(self.rotation_matrices[0]), [p1[0], p1[1], 0])
        p2_rotated = np.dot(np.linalg.inv(self.rotation_matrices[1]), [p2[0], p2[1], 0])
        p3_rotated = np.dot(np.linalg.inv(self.rotation_matrices[2]), [p3[0], p3[1], 0])

        # Average the results to find the best estimate of the 3D position
        x = (p1_rotated[0] + p2_rotated[0] + p3_rotated[0]) / 3
        y = (p1_rotated[1] + p2_rotated[1] + p3_rotated[1]) / 3
        z = (p1_rotated[2] + p2_rotated[2] + p3_rotated[2]) / 3
        return x, y, z

def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()
def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()
def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()
def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()
def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()
def triangulate_points(self, p1, p2, p3):
    """Combine points from rotated planes into a single 3D point using least squares."""
    # Define each plane's rotation matrix and 2D point
    planes = [
        {"R": self.rotation_matrices[0], "point": p1},
        {"R": self.rotation_matrices[1], "point": p2},
        {"R": self.rotation_matrices[2], "point": p3},
    ]

    # Formulate the system of equations
    A = []
    b = []
    for plane in planes:
        R_inv = np.linalg.inv(plane["R"])
        p_2d = np.array([plane["point"][0], plane["point"][1], 1.0])  # Homogeneous 2D point

        # Line equation in 3D from the plane's perspective
        line_dir = R_inv[:, 2]  # Plane's normal vector in 3D
        line_point = np.dot(R_inv, p_2d)  # Point on the line in 3D

        # Add constraints for least squares
        A.append(line_dir)
        b.append(np.dot(line_dir, line_point))

    # Solve using least squares
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    point_3d, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return point_3d.flatten()


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
