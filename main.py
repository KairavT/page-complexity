import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

test = cv.imread('images/IMG_8262.jpg')
test_grey = cv.cvtColor(test, cv.COLOR_BGR2GRAY)

#plt.imshow(test)
#plt.show()
#plt.imshow(test_grey, cmap="gray")
#plt.show()

test_blurred = cv.GaussianBlur(test_grey, (5,5), sigmaX=0)
#plt.imshow(test_blurred, cmap="gray")
#plt.show()
test_canny = cv.Canny(test_blurred, 50, 150)
#plt.imshow(test_canny,cmap="gray")
#plt.show()

kernel = np.ones((5,5), np.uint8)
test_dilated = cv.dilate(test_canny, kernel)

test_contours, _ = cv.findContours(test_dilated, 
                                 cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_NONE)

test_sorted = sorted(test_contours,
                    key = cv.contourArea,
                    reverse=True)

#print(len(test_sorted), cv.contourArea(test_sorted[0]))

test_copy = test.copy()

test_contoured = cv.drawContours(test_copy, test_sorted,
                                 0, color = (230, 30, 0),
                                 thickness=5)
#plt.imshow(test_contoured)
#plt.show()

test_arclen = cv.arcLength(test_sorted[0], True)
test_approxcurve = cv.approxPolyDP(test_sorted[0],
                                   0.02* test_arclen,
                                   True)

test_approxcurve = test_approxcurve.reshape(4, 2)

test_sums = test_approxcurve.sum(axis=1)
top_left = test_approxcurve[np.argmin(test_sums)]
bottom_right = test_approxcurve[np.argmax(test_sums)]

test_diffs = np.diff(test_approxcurve, axis=1)
bottom_left = test_approxcurve[np.argmax(test_diffs)]
top_right = test_approxcurve[np.argmin(test_diffs)]


test_rect = np.array([top_left, top_right, bottom_right, bottom_left],
                      dtype=np.float32)

width = int(np.linalg.norm(top_left - top_right))
height = int(np.linalg.norm(top_left - bottom_left))

test_dest = np.array([(0,0), (width,0), (width,height), (0,height)],
                     dtype=np.float32)

test_matrix = cv.getPerspectiveTransform(test_rect, test_dest)

test_warped = cv.warpPerspective(test,
                                 test_matrix,
                                 (width, height))
plt.imshow(test_warped)
plt.show()