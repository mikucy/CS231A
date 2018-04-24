# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('/home/cp612sh/CS231A/ps0_code/image1.jpg', mode='F')
    print(img1.shape)
    plt.figure(1)
    plt.imshow(img1, cmap='gray')
    u, s, vh = np.linalg.svd(img1)
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    rank1approx = np.dot((u[:, 0] * s[:1]).reshape(-1, 1), vh[0, :].reshape(1, -1))
    plt.figure(2)
    plt.imshow(rank1approx, cmap='gray')
    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank20approx = np.dot(u[:, :20] * s[0: 20], vh[0: 20, :])
    plt.figure(3)
    plt.imshow(rank20approx, cmap='gray')
    plt.show()
    plt.close()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()