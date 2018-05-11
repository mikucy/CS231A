# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    #TODO: Fill in this code
    # let c = 1, [a, b].T = [[x1, y1], [x2, y2]].inv.dot([[-1], [-1]])
    l1 = points[:2]
    l2 = points[2:]
    a1 = l1[0, 1] - l1[1, 1]
    b1 = -(l1[0, 0] - l1[1, 0])
    c1 = -(a1*l1[0, 0] + b1*l1[0, 1])
    a2 = l2[0, 1] - l2[1, 1]
    b2 = -(l2[0, 0] - l2[1, 0])
    c2 = -(a2*l2[0, 0] + b2*l2[0, 1])
    l1 = np.array([a1, b1, c1])
    l2 = np.array([a2, b2, c2])
    vanishing_point = np.cross(l1, l2)
    vanishing_point = vanishing_point / vanishing_point[2]
    return vanishing_point
'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    #TODO: Fill in this code
    # v1 * w * v2.T = 0
    # v2 * w * v3.T = 0
    # v3 * w * v1.T = 0
    # This will generate a linear matrix equation of the form A(3,4) * w(4, 1) = 0(3, 1)
    # e.g. [x1x2, x1+x2, y1+y2, 1] * [w1, w2, w3, w4].T = 0
    # Then w can be calculated as the nullspace of A. Finally K can be recovered by a Cholesky factorization as w = (KK.T)^-1
    p1, p2, p3 = vanishing_points[0], vanishing_points[1], vanishing_points[2]
    A = np.zeros((3, 4))
    A[0] = np.array([p1[0]*p2[0]+p1[1]*p2[1], p1[0]+p2[0], p1[1]+p2[1], 1])
    A[1] = np.array([p2[0]*p3[0]+p2[1]*p3[1], p2[0]+p3[0], p2[1]+p3[1], 1])
    A[2] = np.array([p3[0]*p1[0]+p3[1]*p1[1], p3[0]+p1[0], p3[1]+p1[1], 1])
    # compute the nullspace of A
    rank = np.linalg.matrix_rank(A)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    t_v_A = vh.T
    ncols = t_v_A.shape[1]
    w = t_v_A[:, rank:ncols]
    w = np.array([[w[0], 0, w[1]], [0, w[0], w[2]], [w[1], w[2], w[3]]], dtype=np.float64)
    K = np.linalg.inv(np.linalg.cholesky(w)).T
    return K
'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    #TODO: Fill in this code
    L1 = np.cross(vanishing_pair1[0], vanishing_pair1[1])
    L2 = np.cross(vanishing_pair2[0], vanishing_pair2[1])
    w_inv = K.dot(K.T)
    cos = L1.T.dot(w_inv).dot(L2) / np.sqrt(L1.T.dot(w_inv).dot(L1) * L2.T.dot(w_inv).dot(L2))
    angle = np.arccos(cos) * 180 / math.pi
    return angle
'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    #TODO: Fill in this code
    K_inv = np.linalg.inv(K)
    v1 = np.stack(vanishing_points1).T
    v2 = np.stack(vanishing_points2).T
    # d = K^-1 * v / ||K^-1 * v||
    d1 = K_inv.dot(v1) / np.linalg.norm(K_inv.dot(v1))
    d2 = K_inv.dot(v2) / np.linalg.norm(K_inv.dot(v2))
    # d2 = R * d1, d2.T = d1.T * R.T
    R = np.linalg.lstsq(d1.T, d2.T, rcond=None)[0].T
    return R
if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
