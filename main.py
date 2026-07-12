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

print(len(test_sorted), cv.contourArea(test_sorted[0]))

test_copy = test.copy()

test_contoured = cv.drawContours(test_copy, test_sorted,
                                 0, color = (230, 30, 0),
                                 thickness=5)
plt.imshow(test_contoured)
plt.show()
