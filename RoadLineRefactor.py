import cv2
import math
import numpy as np
import logging
import threading
from bresenham import bresenham

class RoadLines:

    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        self._ret, self._frame = self.capture.read()
        self._source_height, self._source_width = self._frame.shape[:2]
        self._current_width, self._current_height = self._source_width, self._source_height
        self._scaled_width, self._scaled_height = None, None
        self._current_frame = self._frame

        self._vertices = None
        self._ROI_Height = None

        self._right_x1, self._right_x2 = None, None
        self._left_x1, self._left_x2 = None, None
        self._left_lines_x, self._left_lines_y = [], []
        self._right_lines_x, self._right_lines_y = [], []

        self._color = None
        self._channel = None
        self._smooth = None

        self._linesExist = False

    def getColorChannel(self, frame, channel):
        return frame[:,:,channel]

    def caclulateLeftPoints(self):
        x1 = self._left_x1
        if x1 < 0:
            x1 = 0
        return list(bresenham(x1,self._current_height, self._left_x2, self._ROI_Height))

    def caclulateRightPoints(self):
        x1 = self._right_x1
        if x1 < 0:
            x1 = 0
        return list(bresenham(x1,self._current_height, self._right_x2, self._ROI_Height))

    def defineRegionOfInterest(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self._ROI_Height = int(y1)
        self._vertices = np.array(
            [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
            dtype=np.int32)

    def scaleFrame(self, width, height, input_frame):
        self._current_width = width
        self._current_height = height
        self._current_frame = cv2.resize(input_frame, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

    def releaseCaptureSource(self):
        self.capture.release()

    def readSourceCapture(self):
        ret, ret_frame = self.capture.read()
        self._current_frame = ret_frame
        return ret, ret_frame

    def getCurrentFrame(self):
        return self._current_frame

    def convertToGrey(self):
        return cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2GRAY)

    def convertToHLS(self):
        return cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2HLS)

    def convertToXYZ(self):
        return cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2XYZ)

    def convertToLAB(self):
        return cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2LAB)

    def getEdges(self, frame, low_threshold = 50, high_threshold = 100):
        return cv2.Canny(frame, low_threshold, high_threshold)

    def smoothFrame(self, frame, kernel_size=15):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def getOrignalHeight(self):
        return self._source_height

    def getOriginalWidth(self):
        return self._source_width

    def getHougLines(self, frame):

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 0  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        return lines

    def getRegionOfInterest(self, frame):

        # defining a blank mask to start with
        if self._vertices is None:
            self.defineRegionOfInterest(self._current_width, self._current_height / 1.5,
                                        0, self._current_height / 1.5, 0, self._current_height,
                                        self._current_width, self._current_height)

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

                    #slope_percentage = abs(np.divide(y_diff, x_diff))
                    slope_degree = abs(math.atan(np.divide(y_diff, x_diff))) * 100

                    if slope > 0 and slope_degree > 10:
                        self._right_lines_x.append(x1)
                        self._right_lines_x.append(x2)

                        self._right_lines_y.append(y1)
                        self._right_lines_y.append(y2)

            y1 = self._current_height
            y2 = self._ROI_Height

            if len(self._right_lines_y) > 0:
                right_m = np.polyfit(self._right_lines_x, self._right_lines_y, 1)[0]
                right_b = np.polyfit(self._right_lines_x, self._right_lines_y, 1)[1]

                right_x1 = int((y1 - right_b) / right_m)
                right_x2 = int((y2 - right_b) / right_m)

                if right_x1 > self._current_width / 2:
                    self._right_x1 = right_x1

                if right_x2 > self._current_width / 2:
                    self._right_x2 = right_x2

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

            y1 = self._current_height
            y2 = self._ROI_Height

            if len(self._left_lines_y) > 0:
                left_m = np.polyfit(self._left_lines_x, self._left_lines_y, 1)[0]
                left_b = np.polyfit(self._left_lines_x, self._left_lines_y, 1)[1]

                left_x1 = int((y1 - left_b) / left_m)
                left_x2 = int((y2 - left_b) / left_m)

                if left_x1 < self._current_width / 2:
                    self._left_x1 = left_x1

                if left_x2 < self._current_width / 2:
                    self._left_x2 = left_x2

    def findLines(self):
        return

    def weightedFrame(self, img, alpha=0.8, beta=1., _lambda=0.):
        mask = np.zeros_like(img)

        y1 = self._current_height
        y2 = self._ROI_Height

        if self._left_x1 == None:
            self._left_x1 = 0
        if self._left_x2 == None:
            self._left_x2 = 0

        if self._right_x1 == None:
            self._right_x1 = 0
        if self._right_x2 == None:
            self._right_x2 = 0


        cv2.line(mask, (self._right_x1, y1), (self._right_x2, y2), [0, 255, 0], 5)
        cv2.line(mask, (self._left_x1, y1), (self._left_x2, y2), [0, 255, 0], 5)
        return cv2.addWeighted(img, alpha, mask, beta, _lambda)

rl = RoadLines('test4.mp4')

while True:

    ret, frame = rl.readSourceCapture()
    rl.scaleFrame(rl.getOriginalWidth() / 2, rl.getOrignalHeight() / 2, frame)

    if ret:
        color = rl.convertToLAB()
        test = rl.getColorChannel(rl.convertToHLS(), 2)

        channel = rl.getColorChannel(color, 2)
        smooth = rl.smoothFrame(channel, 5)

        edges = rl.getEdges(channel)

        lines = rl.getHougLines(rl.getRegionOfInterest(rl.getEdges(smooth)))

        rl.calculateLinesRight(lines)
        rl.calculateLinesLeft(lines)

        cv2.imshow("othetest", rl.getRegionOfInterest(rl.getEdges(smooth)))
        cv2.imshow("testFrame", rl.weightedFrame(rl.getCurrentFrame()))
        cv2.imshow("orig", edges)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

rl.capture.release()
cv2.destroyAllWindows()