import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from imageio import imread
from utils import *

'''
RUN_DETECTOR Given an image, runs the SVM detector and outputs bounding
boxes and scores

Arguments:
    im - the image matrix

    clf - the sklearn SVM object. You will probably use the
        decision_function() method to determine whether the object is
        a face or not.
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    window_size - an array which contains the height and width of the sliding
    	window

    cell_size - each cell will be of size (cell_size, cell_size) pixels

    block_size - each block will be of size (block_size, block_size) cells

    nbins - number of histogram bins

Returns:
    bboxes - D x 4 bounding boxes that tell [xmin ymin width height] per bounding
    	box

    scores - the SVM scores associated with each bounding box in bboxes

You can compute the HoG features using the compute_hog_features() method
that you implemented in PS3. We have provided an implementation in utils.py,
but feel free to use your own implementation. You will use the HoG features
in a sliding window based detection approach.

Recall that using a sliding window is to take a certain section (called the
window) of the image and compute a score for it. This window then "slides"
across the image, shifting by either n pixels up or down (where n is called
the window's stride).

Using a sliding window approach (with stride of block_size * cell_size / 2),
compute the SVM score for that window. If it's greater than 1 (the SVM decision
boundary), add it to the bounding box list. At the very end, after implementing
nonmaximal suppression, you will filter the nonmaximal bounding boxes out.
'''
def run_detector(im, clf, window_size, cell_size, block_size, nbins, thresh=1):
    # TODO: Implement this method!
    H, W = im.shape
    width, height = window_size[1], window_size[0]
    step = cell_size * block_size // 2
    bboxes = []
    scores = []
    for i in range(0, H - height + step, step):
        for j in range(0, W - width + step, step):
            patch = im[i: i + height, j : j + width]
            features = compute_hog_features(patch, cell_size, block_size, nbins)
            face_decision = clf.decision_function(features.reshape(1, -1))
            if face_decision > thresh:
                bbox = np.array([j, i, width, height])
                bboxes.append(bbox)
                scores.append(face_decision)
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    return bboxes, scores
'''
NON_MAX_SUPPRESSION Given a list of bounding boxes, returns a subset that
uses high confidence detections to suppresses other overlapping
detections. Detections can partially overlap, but the
center of one detection can not be within another detection.

Arguments:
    bboxes - ndarray of size (N,4) where N is the number of detections,
        and each row is [x_min, y_min, width, height]

    confidences - ndarray of size (N, 1) of the SVM confidence of each bounding
    	box.


Returns:
    nms_bboxes -  ndarray of size (N, 4) where N is the number of non-overlapping
        detections, and each row is [x_min, y_min, width, height]. Each bounding box
        should not be overlapping significantly with any other bounding box.

In order to get the list of maximal bounding boxes, first sort bboxes by
confidences. Then go through each of the bboxes in order, adding them to
the list if they do not significantly overlap with any already in the list.
A significant overlap is if the center of one bbox is in the other bbox.
'''
def non_max_suppression(bboxes, confidences):
    # TODO: Implement this method!
    indices = np.argsort(-confidences.reshape(1, -1)).flatten()
    N = indices.shape[0]
    nms_bboxes = []
    for i in range(N):
        bbox = bboxes[indices[i], :]
        if i == 0:
            nms_bboxes.append(bbox)
        else:
            isValid = True
            xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            xc = (xmin + (xmin+w)) / 2.0
            yc = (ymin + (ymin+h)) / 2.0
            for j in range(len(nms_bboxes)):
                _xmin, _ymin, _w, _h = nms_bboxes[j][0], nms_bboxes[j][1], nms_bboxes[j][2], nms_bboxes[j][3]
                _xmax, _ymax = (_xmin + _w), (_ymin + _h)
                if (_xmin <= xc <= _xmax) and (_ymin <= yc <= _ymax):
                    isValid = False
                    break

            if isValid:
                nms_bboxes.append(bbox)
    return np.array(nms_bboxes)


if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([36, 36])

    # compute or load features for training
    if not (os.path.exists('./ps4_code/data/features_pos.npy') and os.path.exists('./ps4_code/data/features_neg.npy')):
        features_pos = get_positive_features('./ps4_code/data/caltech_faces/Caltech_CropFaces', cell_size, window_size, block_size, nbins)
        num_negative_examples = 10000
        features_neg = get_random_negative_features('./ps4_code/data/train_non_face_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('./ps4_code/data/features_pos.npy', features_pos)
        np.save('./ps4_code/data/features_neg.npy', features_neg)
    else:
        features_pos = np.load('./ps4_code/data/features_pos.npy')
        features_neg = np.load('./ps4_code/data/features_neg.npy')

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)

    # Part A: Sliding window detector
    im = imread('./ps4_code/data/people.jpg').astype(np.uint8)
    bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins)
    plot_img_with_bbox(im, bboxes, 'Without nonmaximal suppresion')
    plt.show()

    # Part B: Nonmaximal suppression
    bboxes = non_max_suppression(bboxes, scores)
    plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
    plt.show()
