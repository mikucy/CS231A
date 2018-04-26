# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('./ps0_code/image1.jpg')
    img2 = misc.imread('./ps0_code/image2.jpg')
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1.astype(float)
    img2.astype(float)
    img1nom = img1 / np.max(img1)
    img2nom = img2 / np.max(img2)
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    imgAdd = img1 + img2
    imgAddNorm = imgAdd / np.max(imgAdd)
    plt.figure('Problem3_1')
    plt.imshow(imgAddNorm)
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    imgSize = img1.shape
    mid = imgSize[1] // 2
    newImage1 = np.concatenate((img1[:, :mid, :], img2[:, mid:, :]), axis=1)
    plt.figure('Problem3_2')
    plt.imshow(newImage1)
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros(imgSize, dtype=np.int64)
    for i in range(imgSize[0]):
        if i % 2 == 0:
            newImage2[i, :, :] = img1[i, :, :]
        else:
            newImage2[i, :, :] = img2[i, :, :]
    plt.figure('Problem3_3')
    plt.imshow(newImage2)
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    img1_reshaped = img1.reshape(imgSize[0]//2, imgSize[1]*2, -1)
    img2_reshaped = img2.reshape(imgSize[0]//2, imgSize[1]*2, -1)
    newImage3 = np.concatenate((img1_reshaped[:, :imgSize[1], :], img2_reshaped[:, imgSize[1]:, :]), axis=1)
    newImage3 = newImage3.reshape(imgSize)
    plt.figure('Problem3_4')
    plt.imshow(newImage3)
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    gray = np.dot(newImage3[..., :3], [0.299, 0.587, 0.114])
    fig3 = plt.figure('Problem3_5')
    fig3.suptitle('gray image', fontsize=20)
    
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.show()
    # END YOUR CODE HERE
    plt.close()

if __name__ == '__main__':
    main()