# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()