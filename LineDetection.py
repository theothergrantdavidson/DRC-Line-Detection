import numpy as np
import cv2
from collections import deque
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Capture video from file
cap = cv2.VideoCapture("test.mp4")


def color_selection(image):

    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white_color = cv2.inRange(hls_image, np.uint8([0,100,100]), np.uint8([50,255,255]))
    yellow_color = cv2.inRange(hls_image, np.uint8([20,0,180]), np.uint8([255,80,255]))

    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    return combined_color_images

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def convert_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def get_hist(image):
    hist, bins = np.histogram(convert_grey(res).flatten(), 256, [0, 256])
    return hist.argmax(axis=0)

def eq_Hist(img): # Histogram normalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5, 5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

def lin_img(img, s=1.0, m=0.0):  # Compute linear image transformation img*s+m
    img2 = cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

def contr_img(img, s=1.0):  # Change image contrast; s>1 - increase
    m = 127.0 * (1.0 - s)
    return lin_img(img, s, m)

def region_of_interest(frame):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with

    height, width = frame.shape[:2]
    vertices = np.array(
        [[[0, height / 1.5], [width, height / 1.5], [width, height], [0, height]]],
        dtype=np.int32)
    mask = np.zeros_like(frame)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image

def percieved_illumination(image):
    b, g, r = cv2.split(image)
    return (0.299*r + 0.587*g + 0.114*b)

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    h, w = img.shape[:2]
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = []
    y_min = h
    y_max = int(h * 0.611)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((y2 - y1) / (x2 - x1)) < 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                left_x1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
                left_x2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
            # cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
            elif ((y2 - y1) / (x2 - x1)) > 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                right_x1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
                right_x2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
                #           cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(np.nanmean(right_x1))
    r_avg_x2 = np.int(np.nanmean(right_x2))
    #     print([l_avg_x1, l_avg_x2, r_avg_x1, r_avg_x2])
    cv2.line(img, (l_avg_x1, y_min), (l_avg_x2, y_max), color, thickness)
    cv2.line(img, (r_avg_x1, y_min), (r_avg_x2, y_max), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = img.shape
    draw_lines(line_img, lines)
    return line_img


def get_lines(img, blur_pixels=5, canny_threshold1=100, canny_threshold2=130,
              rho=2, theta=.02, min_line_length=80, max_gap=20, hough_threshold=9):
    # print(rho, theta, min_line_length, max_gap, hough_threshold)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (blur_pixels, blur_pixels))
    img_canny = cv2.Canny(img_blur, canny_threshold1, canny_threshold2)
    lines = cv2.HoughLinesP(img_canny, rho, theta, hough_threshold, min_line_length, max_gap)

    if lines is not None:
        lines = lines.reshape((lines.shape[0], 2, 2))
        lines = lines.astype(float)
    return lines


def line_length(arr):
    l = math.sqrt((arr[0, 0] - arr[1, 0]) ** 2 + (arr[0, 1] - arr[1, 1]) ** 2)
    return l


def line_angle(arr):
    dx = arr[1, 0] - arr[0, 0]
    dy = arr[1, 1] - arr[0, 1]
    rads = math.atan2(-dy, dx)
    rads %= 2 * math.pi
    degs = -math.degrees(rads)
    if degs <= -180:
        degs = degs + 180

    degs = degs + 90
    return degs


def compute_lines(lines):
    from operator import itemgetter

    line_data = []
    for line in lines:
        line_data.append([line_angle(line), line_length(line)])

    sorted(line_data, key=itemgetter(0))
    return line_data

def cluster_angles(line_data):
    clusters = []
    last_angle = -180
    for a, l in line_data:
        if abs(last_angle - a) > 20:
            clusters.append([(a,l)])
        else:
            clusters[-1].append((a,l))
    return clusters

def decide_angle(clustered_angles):
    max_length = 0
    max_cluster_id = -1
    for i, c in enumerate(clustered_angles):
        #sum lenght of lines found in clusters, filter out angles > 80 (likely in horizon)
        cluster_length = sum([l for a, l in c if abs(a) < 80])
        #print('cluster length', cluster_length)
        if cluster_length > max_length:
            max_length = cluster_length
            max_cluster_id = i

    if max_cluster_id>-1:
        angles = [a for a, l in clustered_angles[max_cluster_id]]
        #return average angle of cluster
        return sum(angles)/len(angles)
    #print(angles)
    else:
        return 0

while True:

    ret, frame = cap.read()
    height, width = frame.shape[:2]
    res = cv2.resize(frame,(width / 2, height / 2), interpolation=cv2.INTER_CUBIC)

    if ret == True:

        xyz = cv2.cvtColor(res, cv2.COLOR_BGR2XYZ)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        ret, threshold2 = cv2.threshold(convert_grey(xyz), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        test_xyz = convert_grey(xyz)



        ret, threshold2 = cv2.threshold(test_xyz, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        moving_average = abs(ret - 255)

        edges = detect_edges(xyz, moving_average, 255)

        minLineLength = 50
        maxLineGap = 10
        lines = compute_lines(get_lines(xyz))
        clust_ang = cluster_angles(lines)
        dec_angle = decide_angle(clust_ang)

        print abs(lines)

        print dec_angle



        cv2.imshow("this", res)
        cv2.imshow("other", edges)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

