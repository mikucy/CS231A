# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    N = real_XY.shape[0]
    Z1 = np.zeros((N, 1))
    ones = np.ones((N, 1))
    Z2 = 150 * np.ones((N, 1))
    M_front_scene = np.c_[real_XY, Z1, ones]
    M_back_scene = np.c_[real_XY, Z2, ones]
    M_scene = np.r_[M_front_scene, M_back_scene]
    M_front_image = np.c_[front_image, ones]
    M_back_image = np.c_[back_image, ones]
    M_image = np.r_[M_front_image, M_back_image]
    # Solve for M_scene * camera_matrix.T = M_image
    camera_matrix = np.linalg.lstsq(M_scene, M_image, rcond=None)
    camera_matrix = camera_matrix[0].T
    return camera_matrix

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    #TODO: Fill in this code
    N = real_XY.shape[0]
    Z1 = np.zeros((N, 1))
    ones = np.ones((N, 1))
    Z2 = 150 * np.ones((N, 1))
    M_front_scene = np.c_[real_XY, Z1, ones]
    M_back_scene = np.c_[real_XY, Z2, ones]
    M_scene = np.r_[M_front_scene, M_back_scene]
    M_front_image = np.c_[front_image, ones]
    M_back_image = np.c_[back_image, ones]
    M_image = np.r_[M_front_image, M_back_image]
    predicted = M_scene.dot(camera_matrix.T)
    rms_error = np.sqrt(np.sum(np.square(predicted - M_image)) / N)
    return rms_error

if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('./ps1_code/real_XY.npy')
    front_image = np.load('./ps1_code/front_image.npy')
    back_image = np.load('./ps1_code/back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
