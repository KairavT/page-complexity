import cv2 as cv
import matplotlib.pyplot as plt

test = cv.imread('images/IMG_8262.jpg')
test_grey = cv.cvtColor(test, cv.COLOR_BGR2GRAY)

plt.imshow(test)
plt.show()
plt.imshow(test_grey, cmap="gray")
plt.show()