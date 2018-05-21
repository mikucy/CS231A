# CS231A
My own solutions for CS231A_1718fall problem sets

The course is CS231A: Computer Vision, From 3D Reconstruction to Recognition
<br>The repository contains my solutions for problem sets
<br>All the material can be downloaded from [class syllabus on stanford](http://web.stanford.edu/class/cs231a/syllabus.html)
<br>There are some detailed comments in the codes which I hope will help you understand the program
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

 | Problem3_1 | Problem3_2 | Problem3_3 |
 :---------:|:----------:|:---------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_2.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_3.png) 
 
  | Problem3_4 | Problem3_5 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_4.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem3_5.png)

 - Problem 4

 | Problem4_1 | Problem4_2 | Problem4_3
 :-----:|:-----:|:-----:
 ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_2.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps0_code/solutions/Problem4_3.png)
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
## Problem set 2
 - Problem 1
 ```
 --------------------------------------------------------------------------------
 Set: data/set2
 --------------------------------------------------------------------------------
 Fundamental Matrix from LLS  8-point algorithm:
 [[-5.63087200e-06  2.74976583e-05 -6.42650411e-03]
 [-2.77622828e-05 -6.74748522e-06  1.52182033e-02]
 [ 1.07623595e-02 -1.22519240e-02 -9.99730547e-01]]
 Distance to lines in image 1 for LLS: 9.701438829436539
 Distance to lines in image 2 for LLS: 14.568227190498169
 p'^T F p = 0.033136691062293
 Fundamental Matrix from normalized 8-point algorithm:
 [[-1.51007608e-07  2.51618737e-06 -1.56134009e-04]
 [ 3.63462620e-06  3.22311660e-07  7.02588719e-03]
 [ 2.36155133e-04 -8.53003408e-03 -2.45880925e-03]]
 Distance to lines in image 1 for normalized: 0.8895134540568762
 Distance to lines in image 2 for normalized: 0.8917343723800133
 ```
 | Problem1_1 | Problem1_2 |
 :----:|:----:
 ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem1_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem1_2.png)

  | Problem1_3 | Problem1_4 |
 :----:|:----:
 ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem1_3.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem1_4.png)
 - Problem 2
 ```
 e1 [-1.30071143e+03 -1.42448272e+02  1.00000000e+00]
 e2 [1.65412463e+03 4.53021078e+01 1.00000000e+00]
 H1:
 [[-1.20006316e+01 -4.15501447e+00 -1.23476881e+02]
 [ 1.41006481e+00 -1.48704147e+01 -2.84177469e+02]
 [-9.21889298e-03 -2.19184511e-03 -1.23033440e+01]]
 H2:
 [[ 8.09798131e-01 -1.22036874e-01  7.99331183e+01]
 [-3.00186699e-02  1.01581538e+00  3.63604348e+00]
 [-6.99360915e-04  1.05393946e-04  1.15205554e+00]]
 ```
 ![Problem2](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem2.png)
 - Problem 3

 | Problem3_1 | Problem3_2 |
 :----:|:----:
 ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem3_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem3_2.png)
 - Problem 4
```
--------------------------------------------------------------------------------
Part A: Check your matrices against the example R,T
--------------------------------------------------------------------------------
Example RT:
 [[ 0.9736 -0.0988 -0.2056  0.9994]
 [ 0.1019  0.9948  0.0045 -0.0089]
 [ 0.2041 -0.0254  0.9786  0.0331]]
Estimated RT:
 [[[ 0.98305251 -0.11787055 -0.14040758  0.99941228]
  [-0.11925737 -0.99286228 -0.00147453 -0.00886961]
  [-0.13923158  0.01819418 -0.99009269  0.03311219]]

 [[ 0.98305251 -0.11787055 -0.14040758 -0.99941228]
  [-0.11925737 -0.99286228 -0.00147453  0.00886961]
  [-0.13923158  0.01819418 -0.99009269 -0.03311219]]

 [[ 0.97364135 -0.09878708 -0.20558119  0.99941228]
  [ 0.10189204  0.99478508  0.00454512 -0.00886961]
  [ 0.2040601  -0.02537241  0.97862951  0.03311219]]

 [[ 0.97364135 -0.09878708 -0.20558119 -0.99941228]
  [ 0.10189204  0.99478508  0.00454512  0.00886961]
  [ 0.2040601  -0.02537241  0.97862951 -0.03311219]]]
--------------------------------------------------------------------------------
Part B: Check that the difference from expected point
is near zero
--------------------------------------------------------------------------------
Difference:  0.0029243053036863698
--------------------------------------------------------------------------------
Part C: Check that the difference from expected error/Jacobian
is near zero
--------------------------------------------------------------------------------
Error Difference:  8.301300130674275e-07
Jacobian Difference:  1.817115702351657e-08
--------------------------------------------------------------------------------
Part D: Check that the reprojection error from nonlinear method
is lower than linear method
--------------------------------------------------------------------------------
Linear method error: 98.73542356894195
Nonlinear method error: 95.59481784846034
--------------------------------------------------------------------------------
Part E: Check your matrix against the example R,T
--------------------------------------------------------------------------------
Example RT:
 [[ 0.9736 -0.0988 -0.2056  0.9994]
 [ 0.1019  0.9948  0.0045 -0.0089]
 [ 0.2041 -0.0254  0.9786  0.0331]]
Estimated RT:
 [[ 0.97364135 -0.09878708 -0.20558119  0.99941228]
 [ 0.10189204  0.99478508  0.00454512 -0.00886961]
 [ 0.2040601  -0.02537241  0.97862951  0.03311219]]
```
![Problem4](https://github.com/mikucy/CS231A/raw/master/ps2_code/solutions/Problem4.png)
## Problem Set 3
 - Problem 1

 | Problem1_1 | Problem1_2 | Problem1_3 |
 :---------:|:----------:|:---------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem1_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem1_2.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem1_3.png) 
 
  | Problem1_4 | Problem1_5 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem1_4.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem1_5.png)
 - Problem 2

  | Problem2_1 | Problem2_2 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_2.png)

   | Problem2_3 | Problem2_4 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_3.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_4.png)

  | Problem2_5 | Problem2_6 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_5.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_6.png)

  | Problem2_7 | Problem2_8 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_7.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_8.png)

  | Problem2_9 | Problem2_10 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_9.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_10.png)

  | Problem2_11 | Problem2_12 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_11.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem2_12.png)
 - Problem 3
 ```
 --------------------------------------------------------------------------------
Part A: Image gradient
--------------------------------------------------------------------------------
Expected angle: 126.339396329
Expected magnitude: 0.423547566786
Checking gradient test case 1: True
Expected angles: 
 [[100.30484647  63.43494882 167.47119229]
 [ 68.19859051   0.          45.        ]
 [ 53.13010235  64.53665494 180.        ]]
Expected magnitudes: 
 [[11.18033989 11.18033989  9.21954446]
 [ 5.38516481 11.          7.07106781]
 [15.         11.62970335  2.        ]]
Checking gradient test case 2: True
--------------------------------------------------------------------------------
Part B: Histogram generation
--------------------------------------------------------------------------------
Checking histogram test case 1: True
Checking histogram test case 2: True
Submit these results: [4.535 2.465 0.95  0.8   0.45  0.9   0.    0.    0.   ]
 ```
![Problem3](https://github.com/mikucy/CS231A/raw/master/ps3_code/solutions/Problem3.png)
## Problem Set 4
 - Problem 1

 | Problem1_1 | Problem1_2 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem1_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem1_2.png)
 - Problem 2

  | Problem2_1 | Problem2_2 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_1.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_2.png)

   | Problem2_3 | Problem2_4 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_3.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_4.png)

  | Problem2_5 | Problem2_6 |
  :-------:|:-------:
 ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_5.png) | ![](https://github.com/mikucy/CS231A/raw/master/ps4_code/solutions/Problem2_6.png)
