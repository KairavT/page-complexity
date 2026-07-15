import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



def clean_page(path):
    img = cv.imread(path)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(img_grey, (5,5), sigmaX=0)
    canny = cv.Canny(blurred, 50, 150)

    kernel = np.ones((5,5), np.uint8)
    dilated = cv.dilate(canny, kernel)

    contours, _ = cv.findContours(dilated, 
                                    cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_NONE)

    contours_sorted = sorted(contours,
                        key = cv.contourArea,
                        reverse=True)

    img_copy = img.copy()

    arclen = cv.arcLength(contours_sorted[0], True)
    approxcurve = cv.approxPolyDP(contours_sorted[0],
                                    0.02* arclen,
                                    True)

    
    if len(approxcurve) != 4:
        raise ValueError(f'Incorrect number of corners found in image, '
                        f'{len(approxcurve)} instead of 4')


    approxcurve = approxcurve.reshape(4, 2)
    
        

    sums = approxcurve.sum(axis=1)
    top_left = approxcurve[np.argmin(sums)]
    bottom_right = approxcurve[np.argmax(sums)]

    diffs = np.diff(approxcurve, axis=1)
    bottom_left = approxcurve[np.argmax(diffs)]
    top_right = approxcurve[np.argmin(diffs)]


    rect = np.array([top_left, top_right, bottom_right, bottom_left],
                        dtype=np.float32)

    width = int(np.linalg.norm(top_left - top_right))
    height = int(np.linalg.norm(top_left - bottom_left))

    dest = np.array([(0,0), (width,0), (width,height), (0,height)],
                        dtype=np.float32)

    matrix = cv.getPerspectiveTransform(rect, dest)

    warped = cv.warpPerspective(img,
                                    matrix,
                                    (width, height))

    img_bw = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    img_binary = cv.adaptiveThreshold(
        src=img_bw,
        maxValue=255, 
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=17,
        C=11
    )

    return img_binary

pics = ['images/IMG_8262.jpg', 'images/IMG_8285.jpg',
        'images/IMG_8286.jpg', 'images/IMG_8289.jpg']

for path in pics:
    try:
        result = clean_page(path)
        plt.imshow(result, cmap='gray')
        plt.show()
    except ValueError as e:
        print(path, e)