import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u1_p = points2[:, 0]
    v1_p = points2[:, 1]
    one = np.ones_like(u1)
    W = np.c_[u1 * u1_p, v1 * u1_p, u1_p, u1 * v1_p, v1 * v1_p, v1_p, u1, v1, one]
    # Use svd to find the lstsq solution for Wf = 0
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    f = vh[-1, :]
    F_t = f.reshape(3, 3)
    # Enforce F_t to F which is rank 2
    u, s, vh = np.linalg.svd(F_t, full_matrices=True)
    s[-1] = 0
    F = u.dot(np.diag(s)).dot(vh)
    return F
    # NOTICE!! Here we find the F for p'Fp = 0 instead of pFp' = 0

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    # Find the transformation matrix T and T': [[S, S*t], [0, 1]], because we translate first, then scale
    # Translate: origin as image points centroid
    # Scale: 2 / mean squared distance
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    
    scale1 = np.sqrt(2 / np.mean(np.sum((points1 - mean1) ** 2, axis=1)))
    scale2 = np.sqrt(2 / np.mean(np.sum((points2 - mean2) ** 2, axis=1)))
    T = np.array([[scale1, 0, -scale1 * mean1[0]], [0, scale1, -scale1 * mean1[1]], [0, 0 ,1]])
    T_p = np.array([[scale2, 0, -scale2 * mean2[0]], [0, scale2, -scale2 * mean2[1]], [0, 0, 1]])
    # q = T * p
    points1 = T.dot(points1.T).T
    points2 = T_p.dot(points2.T).T
    Fq = lls_eight_point_alg(points1, points2)
    # de-normalize
    return T_p.T.dot(Fq).dot(T)
'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
    N = points1.shape[0]
    h1 = im1.shape[0]
    w1 = im1.shape[1]
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    # l = F.T * p' is the epipolar line associated with p'
    # l' = F * p is the epipolar line associated with p
    l1 = F.T.dot(points2.T)
    l2 = F.dot(points1.T)
    plt.subplot(121)
    plt.imshow(im1, cmap='gray')
    for i in range(N):
        # l = [a, b, c] => ax + by + c = 0, y = -(a/b)x - c/b
        X1 = np.arange(0.0, w1, 1)
        Y = -(l1[0, i] / l1[1, i]) * X1 - (l1[2, i] / l1[1, i])
        idx = (Y >= 0) & (Y <= h1)
        X1 = X1[idx]
        Y = Y[idx]
        plt.plot(X1, Y, 'r')
        plt.plot(points1[i, 0], points1[i, 1], '*', color='b')
    plt.subplot(122)
    plt.imshow(im2, cmap='gray')
    for i in range(N):
        # l = [a, b, c] => ax + by + c = 0, y = -(a/b)x - c/b
        X2 = np.arange(0.0, w2, 1)
        Y = -(l2[0, i] / l2[1, i]) * X2 - (l2[2, i] / l2[1, i])
        idx = (Y >= 0) & (Y <= h2)
        X2 = X2[idx]
        Y = Y[idx]
        plt.plot(X2, Y, 'r')
        plt.plot(points2[i, 0], points2[i, 1], '*', color='b')
'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    l = F.T.dot(points2.T)
    # distance from point(x0, y0) to line: Ax + By + C = 0 is
    # |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    d = np.mean(np.abs(np.sum(l * points1.T, axis=0)) / np.sqrt(l[0, :] ** 2 + l[1, :] ** 2))
    return d

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread('./ps2_code/'+im_set+'/image1.jpg')
        im2 = imread('./ps2_code/'+im_set+'/image2.jpg')
        points1 = get_data_from_txt_file('./ps2_code/'+im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file('./ps2_code/'+im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plt.figure(1)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plt.suptitle('epipolar lines for lls')
        plt.figure(2)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)
        plt.suptitle('epipolar lines for normalized')

        plt.show()
