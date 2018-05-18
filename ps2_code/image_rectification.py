import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''

# NOTICE!! (I think the comment is wrong, because in main() it calculates F as p'Fp=0, so I will treat it that way)
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
    # p.'T * F * p = 0
    l1 = F.T.dot(points2.T).T
    # Solve l1 * e = 0
    u, s, vh = np.linalg.svd(l1)
    e = vh[-1, :]
    e = e / e[2]
    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    # e' = [e1', e2', 1]
    # H2 = T^-1 * G * R * T
    # T = [1, 0, -width/2]   G = [1, 0, 0]     R = [α*e1'/sqrt(e1'^2+e2'^2), α*e2'/sqrt(e1'^2+e2'^2), 0]
    #     [0, 1, -height/2]      [0, 1, 0]         [-α*e2'/sqrt(e1'^2+e2'^2), α*e1'/sqrt(e1'^2+e2'^2), 0]
    #     [0, 0, 1]              [-1/f, 0, 1]      [0, 0, 1]
    h = im2.shape[0]
    w = im2.shape[1]
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
    e2_p = T.dot(e2)
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e2_p = R.dot(e2_p)
    f = e2_p[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/f, 0, 1]])
    H2 = np.linalg.inv(T).dot(G).dot(R).dot(T)
    
    # H1 = H_A * H2 * M
    # H_A = [a1, a2, a3]  M = e_x * F + e * [1, 1, 1]  e'_x = [0, -z, y]
    #       [0, 1, 0]                                        [z, 0, -x]
    #       [0, 0, 1]                                        [-y, x, 0]
    # a1, a2, a3 is the solution for Wa = b, where
    # W = [[x1~, y1~, 1], [x2~, y2~, 1], ...] b = [x1'~, x2'~, ... xn'~]
    e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = e_x.dot(F) + e2.reshape(3,1).dot(np.array([[1, 1, 1]]))
    points1_t = H2.dot(M).dot(points1.T)
    points2_t = H2.dot(points2.T)
    points1_t /= points1_t[2, :]
    points2_t /= points2_t[2, :]
    b = points2_t[0, :]
    a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
    H_A = np.array([a, [0, 1, 0], [0, 0, 1]])
    H1 = H_A.dot(H2).dot(M)
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread('./ps2_code/'+im_set+'/image1.jpg')
    im2 = imread('./ps2_code/'+im_set+'/image2.jpg')
    points1 = get_data_from_txt_file('./ps2_code/'+im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file('./ps2_code/'+im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
