import cv2
import numpy as np
import math
from bresenham import bresenham
import threading
import time


class RoadLine():

    def __init__(self, source):
        import logging as log

        self.log = log
        self.log.basicConfig(filename='RoadLineError.log',level=log.DEBUG)
        self.cap = cv2.VideoCapture(source)
        rets, frame = self.cap.read()
        self._h, self._w = frame.shape[:2]
        self._frame = None
        self._colourd_frame = None
        self._vertices = None
        self._ROI_y1, ROI_y2 = None, None
        self._right_x1, self._right_x2 = None, None
        self._left_x1, self._left_x2 = None, None
        self._current_h , self._current_w = None, None
        self._left_lines_x, self._left_lines_y = [], []
        self._right_lines_x, self._right_lines_y = [], []

    def caclulateLeftPoints(self):
        x1 = self._left_x1
        if x1 < 0:
            x1 = 0
        return list(bresenham(x1,self._current_h, self._left_x2, self._ROI_y1))

    def caclulateRightPoints(self):
        x1 = self._right_x1
        if x1 < 0:
            x1 = 0
        return list(bresenham(x1,self._current_h, self._right_x2, self._ROI_y1))

    def readSourceFrame(self):
        ret, ret_frame = self.cap.read()
        self._frame = ret_frame
        return ret, ret_frame

    def smoothFrame(self, frame, kernel_size=15):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def getSourceFrame(self):
        return self._frame

    def scaleFrame(self, w, h):
        self._current_w = int(w)
        self._current_h = int(h)
        self._frame = cv2.resize(self._frame, (w, h), interpolation=cv2.INTER_CUBIC)

    def getOrignalSize(self):
        return self._w, self._h

    def getCurrentFrameSize(self, frame):
        height, width = frame.shape[:2]
        return width, height

    def convertToGrey(self):
        frame = self._frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def convertToHLS(self):
        frame = self._frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    def convertToXYZ(self):
        frame = self._frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)

    def convertToLAB(self):
        frame = self._frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    def getEdges(self, frame,low_threshold, high_threshold):
        return cv2.Canny(frame, low_threshold, high_threshold)

    def getColorChannel(self, frame, channel):
        return frame[:,:,channel]

    def getConvertedFrame(self):
        return self._cvtFrame

    def getMaskedImage(self):
        return self._masked_image

    def getCurrentFrame(self):
        return self._frame

    def defineRegionOfInterest(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self._ROI_y1 = int(y1)
        self._ROI_y2 = int(y3)
        self._vertices = np.array(
            [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
            dtype=np.int32)

    def getHougLines(self, frame):

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 0  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        return lines

    def getRegionOfInterest(self, frame):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        if self._vertices is None:
            self.log.error("Region of interest has not been defined before getting region of interest of a frame")
            raise ValueError("Region of interest has not been defined before getting region of interest of a frame")
        else :
            mask = np.zeros_like(frame)

            # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
            if len(frame.shape) > 2:
                channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,) * channel_count
            else:
                ignore_mask_color = 255

            # filling pixels inside the polygon defined by "vertices" with the fill color
            cv2.fillPoly(mask, self._vertices, ignore_mask_color)

            # returning the image only where mask pixels are nonzero
            masked_image = cv2.bitwise_and(frame, mask)
            return masked_image

    def calculateLinesRight(self, lines):
        # slope calculation m = (y2 - y1) / (x2 - x1)
        # slope degree of slope = tan-1((y2 - y1) / (x2 - x1))
        # percentage of slope = ((y2 - y1) / (x2 - x1)) * 100

        if lines is not None and len(lines) > 0:

            self._right_lines_x = []
            self._right_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff)

                    slope_percentage = abs(np.divide(y_diff, x_diff))
                    slope_degree = abs(math.atan(np.divide(y_diff, x_diff))) * 100

                    if slope > 0 and slope_degree > 10:
                        self._right_lines_x.append(x1)
                        self._right_lines_x.append(x2)

                        self._right_lines_y.append(y1)
                        self._right_lines_y.append(y2)

            y1 = self._current_h
            y2 = self._ROI_y1

            if len(self._right_lines_y) > 0:
                right_m = np.polyfit(self._right_lines_x, self._right_lines_y, 1)[0]
                right_b = np.polyfit(self._right_lines_x, self._right_lines_y, 1)[1]

                self._right_x1 = int((y1 - right_b) / right_m)
                self._right_x2 = int((y2 - right_b) / right_m)

                return True
            else:
                return False
        else:
            return False

    def calculateLinesLeft(self, lines):
        # slope calculation m = (y2 - y1) / (x2 - x1)
        # slope degree of slope = tan-1((y2 - y1) / (x2 - x1))
        # percentage of slope = ((y2 - y1) / (x2 - x1)) * 100

        if lines is not None and len(lines) > 0:

            self._left_lines_x = []
            self._left_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff)

                    slope_percentage = abs(np.divide(y_diff, x_diff))
                    slope_degree = abs(math.atan(np.divide(y_diff, x_diff))) * 100

                    if slope < 0 and slope_degree > 10:
                        self._left_lines_x.append(x1)
                        self._left_lines_x.append(x2)

                        self._left_lines_y.append(y1)
                        self._left_lines_y.append(y2)

            y1 = self._current_h
            y2 = self._ROI_y1

            if len(self._left_lines_y) > 0:
                left_m = np.polyfit(self._left_lines_x, self._left_lines_y, 1)[0]
                left_b = np.polyfit(self._left_lines_x, self._left_lines_y, 1)[1]

                self._left_x1 = int((y1 - left_b) / left_m)
                self._left_x2 = int((y2 - left_b) / left_m)

                return 0
            else:
                return 1
        else:
            return 1

    def weighted_img(self, img, alpha=0.8, beta=1., _lambda=0.):
        mask = np.zeros_like(img)

        y1 = self._current_h
        y2 = self._ROI_y1

        if self._left_x1 == None:
            self._left_x1 = 0
        if self._left_x2 == None:
            self._left_x2 = 0

        if self._right_x1 == None:
            self._right_x1 = 0
        if self._right_x2 == None:
            self._right_x2 = 0

        top_left_ROI = [self._left_x2, y2]
        top_right_ROI = [self._right_x2, y2]
        bottom_right_ROI = [self._right_x1, y1]
        bottom_left_ROI = [self._left_x1, y1]

        vertices = np.array(
            [[top_left_ROI, top_right_ROI, bottom_right_ROI, bottom_left_ROI]],
            dtype=np.int32)

        cv2.fillPoly(mask, vertices, [230, 100, 0])

        cv2.line(mask, (self._right_x1, y1), (self._right_x2, y2), [0, 255, 0], 5)
        cv2.line(mask, (self._left_x1, y1), (self._left_x2, y2), [0, 255, 0], 5)
        return cv2.addWeighted(img, alpha, mask, beta, _lambda)

    def getRightLines(self):
        return self._right_lines_x, self._right_lines_y

rd = RoadLine(0)
w, h = rd.getOrignalSize()
w_r = w / 2
h_r = h / 2
rd.defineRegionOfInterest(w_r, h_r / 1.5, 0, h_r / 1.5, 0, h_r, w_r, h_r)
leftThreadStop = threading.Event()
rightThreadStop = threading.Event()

while True:
    ret, frame = rd.readSourceFrame()
    rd.scaleFrame(w / 2, h / 2)

    if ret == True:
        start = time.time()
        original = rd.getSourceFrame()
        color = rd.convertToLAB()

        channel = rd.getColorChannel(color, 2)
        canny = rd.getEdges(channel, 50, 150)
        lines = rd.getHougLines(rd.getRegionOfInterest(canny))

        leftThread = threading.Thread(target= rd.calculateLinesLeft, args=(lines,))
        rightThread = threading.Thread(target=rd.calculateLinesRight, args=(lines,))
        #rd.calculateLinesLeft(lines)
        #rd.calculateLinesRight(lines)
        leftThread.start()
        rightThread.start()
        weight = rd.weighted_img(original)
        cv2.imshow("this",weight)
        cv2.imshow("this2", channel)
        end = time.time()
        print(end - start)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

rd.cap.release()
cv2.destroyAllWindows()
