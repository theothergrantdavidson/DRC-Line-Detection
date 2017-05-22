import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2
#define capture sorce
cap = cv2.VideoCapture("test4.mp4")

#do one read to populate size
rets, frame = cap.read()
h, w = frame.shape[:2]
resized_h = h / 2
resized_w = w / 2

#Globals#
right_x1 = 0
right_x2 = 0

left_x1 = 0
left_x2 = 0

prev_left_x1 = 0
prev_left_x2 = 0

prev_right_x1 = 0
prev_right_x2 = 0


#Gaussian Smoothing
kernel_size = 3

#Canny Edge Detector Thresholds
low_threshold = 50
high_threshold = 150

#Region of interest
top_left_ROI = [w * .15, resized_h / 1.5]
top_right_ROI = [resized_w - (resized_w * .30), resized_h / 1.5]
bottom_right_ROI = [resized_w, resized_h]
bottom_left_ROI = [0, resized_h]

vertices = np.array(
        [[top_left_ROI, top_right_ROI, bottom_right_ROI, bottom_left_ROI]],
        dtype=np.int32)

# Hough Transform
rho = 1 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 30	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 0 #minimum number of pixels making up a line
max_line_gap = 10	# maximum gap in pixels between connectable line segments

#Helper functions
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def convert_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def canny(frame, low_threshold = low_threshold, high_threshold = high_threshold):
    return cv2.Canny(frame, low_threshold, high_threshold)

def blur(frame, kernel_size):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def convert_xyz(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)

def apply_smoothing(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices = vertices):

    # create an empty array in the same structure as the img array except with zero for every value
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  #count of how many channels there are
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # fill pixels in mask polygon
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 150,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def draw_line(img, lines):
    #slope calculation m = (y2 - y1) / (x2 - x1)
    #slope degree of slope = tan-1((y2 - y1) / (x2 - x1))
    #percentage of slope = ((y2 - y1) / (x2 - x1)) * 100

    if lines is None:
        return img
    if len(lines) == 0:
        return img

    mask = np.zeros_like(img)

    right_lines_x = []
    right_lines_y = []

    left_lines_x = []
    left_lines_y = []

    draw_right = True
    draw_left = True

    for x1, y1, x2, y2 in lines[0]:
        y_diff = np.float64(y2 - y1)
        x_diff = np.float64(x2 - x1)

        slope  = np.divide(y_diff, x_diff)

        slope_percentage = abs(np.divide(y_diff, x_diff))
        slope_degree = abs(math.atan(np.divide(y_diff , x_diff))) * 100

        if slope > 0 and slope_degree > 10:
            right_lines_x.append(x1)
            right_lines_x.append(x2)

            right_lines_y.append(y1)
            right_lines_y.append(y2)

        if slope < 0 and slope_degree > 10:
            left_lines_x.append(x1)
            left_lines_x.append(x2)

            left_lines_y.append(y1)
            left_lines_y.append(y2)

    right_m, right_b = 1, 1
    left_m, left_b = 1, 1

    y1 = resized_h
    y2 = int(img.shape[0] / 1.5)


    if len(right_lines_y) > 0:
        right_m = np.polyfit(right_lines_x, right_lines_y, 1)[0]
        right_b = np.polyfit(right_lines_x, right_lines_y, 1)[1]

        global right_x1, right_x2
        right_x1 = int((y1 - right_b) / right_m)
        right_x2 = int((y2 - right_b) / right_m)


    if len(left_lines_y) > 0:
        left_m = np.polyfit(left_lines_x, left_lines_y, 1)[0]
        left_b = np.polyfit(left_lines_x, left_lines_y, 1)[1]

        global left_x1, left_x2
        left_x1 = int((y1 - left_b) / left_m)
        left_x2 = int((y2 - left_b) / left_m)


    top_left_ROI = [left_x2, y2]
    top_right_ROI = [right_x2,y2]
    bottom_right_ROI = [right_x1, y1]
    bottom_left_ROI = [left_x1, y1]

    vertices = np.array(
        [[top_left_ROI, top_right_ROI, bottom_right_ROI, bottom_left_ROI]],
        dtype=np.int32)

    cv2.fillPoly(mask, vertices, [230, 100, 0])
    cv2.line(mask, (left_x1, y1), (left_x2, y2), [0, 255, 0], 5)
    cv2.line(mask, (right_x1, y1), (right_x2, y2), [0, 255, 0], 5)

    return weighted_img(img, mask)


def weighted_img(img, initial_img,alpha = 0.8, beta = 1., _lambda = 0.):

    return cv2.addWeighted(initial_img, alpha, img, beta , _lambda)


def hough_lines(img):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines

def filter_colors(image):

    # Filter white pixels
    white_threshold = 150 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return image2

while True:

    ret, frame = cap.read()
    img = cv2.resize(frame,(resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    test_color = convert_hls(img)
    lab = convert_lab(img)

    if rets == True:

        color_mask = select_white_yellow(apply_smoothing(img, 3))

        mask1 = cv2.inRange(test_color, np.uint8([100, 150, 10]), np.uint8([255, 255, 255]))
        mask2 = cv2.inRange(lab, np.uint8([0, 10, 150]), np.uint8([255, 255, 255]))

        mask = cv2.bitwise_or(mask1, mask2)
        cmask = cv2.bitwise_and(img, img, mask=mask1)

        lines = hough_lines(region_of_interest(canny(canny(lab[:,:,2]))))
        lined_image = draw_line(img, lines)

        cv2.imshow('img2',lined_image)
        cv2.imshow('lines', canny(lab[:,:,2]))
        cv2.imshow('tes', cmask)




        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
