import cv2 as cv
import matplotlib.pyplot as plt

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
plt.imshow(test_canny,cmap="gray")
plt.show()