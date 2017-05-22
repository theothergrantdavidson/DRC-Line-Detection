import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Capture video from file
cap = cv2.VideoCapture("Test_Sunny.mp4")
temp = 1

def convertGrey(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def color_selection(image):

    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_color = cv2.inRange(hls_image, np.uint8([20,200,0]), np.uint8([255,255,255]))
    yellow_color = cv2.inRange(hls_image, np.uint8([10,50,100]), np.uint8([100,255,255]))

    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    return cv2.bitwise_and(image, image, mask = combined_color_images)


def region_of_interest(img, frameWidth, frameHeight):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    vertices = np.array(
        [[[0, frameHeight / 1.5], [frameWidth, frameHeight / 1.5], [frameWidth, frameHeight], [0, frameHeight]]],
        dtype=np.int32)
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
lowValue = 0
highValue = 255
firstRun = True

while True:

    ret, frame = cap.read()
    height, width = frame.shape[:2]
    res = cv2.resize(frame,(width / 2, height / 2), interpolation=cv2.INTER_CUBIC)
    normalizedImage = np.array((width, height))
    norm = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)

    if ret == True:
        hls = cv2.cvtColor(res, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)
        lab2 = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


        white = cv2.inRange(hls, np.uint8([0, 220, 0]), np.uint8([255, 255, 255]))


        #result = cv2.bitwise_and(res, res, mask=mask)

        cv2.imshow('lab res', hls)

        cv2.imshow('res', lab)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


'''
minLineLength = 20
maxLineGap = 0
lines = cv2.HoughLinesP(detect_edges(mask_yw_image), 2, np.pi / 20, 15, minLineLength, maxLineGap)
for x in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
'''