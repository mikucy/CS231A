# CS231A
My own solutions for CS231A_1718fall problem sets

The course is CS231A: Computer Vision, From 3D Reconstruction to Recognition
<br>The repository contains my solutions for problem sets
<br>All the material can be downloaded from [class syllabus on stanford](http://web.stanford.edu/class/cs231a/syllabus.html)
<br>
# Reference solutions
Here are some solutions which are easy to check, in case someone needs them.
<br>If you find some mistakes, feel free to tell me.
## Problem set 0
- Problem 2
```
(a^T b)Ma: 
 [ 3  9 15  2]
multiply each row of M element-wise by a: 
 [[1 2 0]
 [4 5 0]
 [7 8 0]
 [0 2 0]]
sorted M: 
 [0 0 0 0 0 1 2 2 4 5 7 8]
 ```
 ![Problem2](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem2.png)
- Problem 3

 ![Problem3_1](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_1.png)
 ![Problem3_2](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_2.png)
 ![Problem3_3](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_3.png)
 ![Problem3_4](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_4.png)
 ![Problem3_5](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_5.png)

 - Problem 4

 ![Problem4_1](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_1.png)
 ![Problem4_2](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_2.png)
 ![Problem4_3](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_3.png)
## Problem set 1
 - Problem 2
 ```
 Camera Matrix:
 [[ 5.31276507e-01 -1.80886074e-02  1.20509667e-01  1.29720641e+02]
 [ 4.84975447e-02  5.36366401e-01 -1.02675222e-01  4.43879607e+01]
 [-1.40079365e-18  4.77048956e-18  1.40946282e-18  1.00000000e+00]]
RMS Error:  1.4054885481541644
 ```
 - Problem 3
 ```
 Intrinsic Matrix:
 [[ 2.87873596e+03 -0.00000000e+00  8.58115141e+02]
 [-0.00000000e+00  2.87873596e+03  1.08694938e+03]
 [-0.00000000e+00 -0.00000000e+00  1.10969448e+00]]
 
Actual Matrix:
 [[2.448e+03 0.000e+00 1.253e+03]
 [0.000e+00 2.438e+03 9.860e+02]
 [0.000e+00 0.000e+00 1.000e+00]]
Angle between floor and box: 90.027361241031
Rotation between two cameras:
 [[ 8.19551249e-01  1.35836471e-01 -3.69928448e-01]
 [ 9.09898356e-02  1.16646489e+00  1.42060135e-01]
 [-7.67565342e-16 -1.22124533e-15  1.01061171e+00]]
Angle around z-axis (pointing out of camera): -9.410931 degrees
Angle around y-axis (pointing vertically): -19.924588 degrees
Angle around x-axis (pointing horizontally): -8.001552 degrees

 ```
