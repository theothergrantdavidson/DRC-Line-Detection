import cv2
import math
import numpy as np

class Lines:

    def __init__(self, source):
        '''
        Constructor for Lines class
        
        :ivar capture: This is the entry point for the video feed from the source
        :ivar ret: Boolean which represents the return state of capture when read, False when video feed is broken
        :ivar frame: Video frame returned from capture when it is read
        :ivar scale: A default scale for the Range of Interest
        :ivar right_x1: Bottom x coordinate for the right hand line
        :ivar right_x2: Top x coordinate for the right hand line
        :ivar left_x1: Bottom x coordinate for the left hand line
        :ivar left_x2: Top x coordinate foe the left hand line
        :ivar current_width: Stores the current width of the scaled frame when method is executed
        :ivar left_line_state: Keeps a count on how many times a left line has not been found
        :ivar right_line_state: Keeps a count on how many times a right line has not been found    
        :param source: This can be a string to the path of a video file or integer with a value of 0 for the default webcam.
        '''
        self.capture = cv2.VideoCapture(source)
        self.ret, self.frame = self.capture.read()
        self.scale = 2
        self.right_x1, self.right_x2 = None, None
        self.left_x1, self.left_x2 = None, None
        self.current_width = None

        self.left_line_state = 0
        self.right_line_state = 0

    def getColorChannel(self, frame, channel):
        '''
        Method takes a frame and returns a specified color channel from that frame
        :param frame: A video frame
        :param channel: An integer from 0 - 2 which represents a color channel
        :return: The seperated color channel in the form of a 2 channel video frame
        '''
        return frame[:,:,channel]

    def scaleFrame(self, width, height, input_frame):
        '''
        Scales a frame to the sepcified size and then returns that frame as well as updating the current_width
        of a frame in the classes variable field
        :param width: An Integer for a desired width
        :param height: An Integer for a desired height
        :param input_frame: A video frame
        :return: A 3-Channel scaled video frame
        '''
        self.current_width = width
        return cv2.resize(input_frame, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

    def releaseCaptureSource(self):
        '''
        Releases capture source when program closes
        '''
        self.capture.release()

    def getOriginalFrame(self):
        '''
        Reads a frame from the capture source
        :ivar ret: Boolean which represents the return state of capture when read, False when video feed is broken
        :ivar frame: A 3-Channel RGB Video frame returned from capture when it is read
        :return: ret, ret_frame in the form of a tuple
        '''
        ret, ret_frame = self.capture.read()
        return ret, ret_frame

    def convertToHSV(self, frame):
        '''
        Converts input frame to HSV color space
        :param frame: A 3-Channel RGB video frame
        :return: A 3-Channel HSV frame
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def convertToGrey(self, frame):
        '''
        Converts a input frame to greyscale
        :param frame: A 3-Channel video frame
        :return: A 2-Channel greyscale frame
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def convertToHLS(self, frame):
        '''
        Converts a input frame to HLS
        :param frame: A 3-Channel RGB frame
        :return: A 3-Channel HLS frame
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    def convertToXYZ(self, frame):
        '''
        Converts a input frame to XYZ
        :param frame: A 3-Channel RGB frame
        :return: A 3-Channel XYZ frame
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)

    def convertToLAB(self, frame):
        '''
        Converts a input frame to LAB
        :param frame: A 3-Channel RGB frame
        :return: A 3-Channel LAB frame
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    def getEdges(self, frame, low_threshold = 50, high_threshold = 100):
        '''
        Detects the edges in a 2-Channel image based on the threshold between which an edge should exist
        :param frame: A 2-Channel greyscale frame from either the convertToGrey() or getColorChannel()
        :param low_threshold: A value from 0 - 255 for the low threshold with which to choose edges
        :param high_threshold: A value from 0 - 255 for the low threshold with which to choose edges
        :return: A 2-Channel black and white frame
        '''
        return cv2.Canny(frame, low_threshold, high_threshold)

    def smoothFrame(self, frame, kernel_size=15):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def convertToLUV(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)

    def getRightHough(self, frame):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments

        width, height = self.getFrameSize(frame)

        vertices = np.array(
            [[[width, height / 1.5], [width / self.scale, height / 1.5], [width / self.scale, height],
              [width, height]]], dtype=np.int32)

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

        lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.pi / 180, min_line_length, max_line_gap)
        return lines, masked_image

    def getFrameSize(self, frame):
        # defining a 3 channel or 1 channel frame
        if len(frame.shape) > 2:
            height, width = frame.shape[:2]
            return width, height
        else:
            height, width = frame.shape
            return width, height

    def getLeftHough(self, frame):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments
        width, height = self.getFrameSize(frame)

        vertices = np.array(
            [[[width / self.scale, height / 1.5], [0, height / 1.5], [0, height], [ width/ self.scale, height]]],
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

        lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.pi / 180, min_line_length, max_line_gap)
        return lines, masked_image


    def calculateLinesRight(self, frame):
        lines = self.getRightHough(frame)[0]
        width, height = self.getFrameSize(frame)

        if lines is not None and len(lines) > 0:

            right_lines_x = []
            right_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff)
                    slope_degree = abs(math.atan(slope)) * 100

                    if slope_degree > 10 and slope > 0:
                        right_lines_x.append(x1)
                        right_lines_x.append(x2)

                        right_lines_y.append(y1)
                        right_lines_y.append(y2)

            y1 = height
            y2 = height / 1.5

            if len(right_lines_y) > 0:
                right_m = np.polyfit(right_lines_x, right_lines_y, 1)[0]
                right_b = np.polyfit(right_lines_x, right_lines_y, 1)[1]

                right_x1 = int((y1 - right_b) / right_m)
                right_x2 = int((y2 - right_b) / right_m)

                if right_x1 > width / self.scale:
                    self.right_x1 = right_x1
                    self.right_line_state = 0
                if right_x2 > width / self.scale:
                    self.right_x2 = right_x2
                    self.right_line_state = 0
        else:
            self.right_line_state += 1

    def calculateLinesLeft(self, frame):
        lines = self.getLeftHough(frame)[0]
        width, height = self.getFrameSize(frame)

        if lines is not None and len(lines) > 0:

            left_lines_x = []
            left_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff)
                    slope_degree = abs(math.atan(slope)) * 100

                    if slope_degree > 10 and slope < 0:
                        left_lines_x.append(x1)
                        left_lines_x.append(x2)
                        left_lines_y.append(y1)
                        left_lines_y.append(y2)

            y1 = height
            y2 = height / 1.5

            if len(left_lines_y) > 0:
                left_m = np.polyfit(left_lines_x, left_lines_y, 1)[0]
                left_b = np.polyfit(left_lines_x, left_lines_y, 1)[1]

                left_x1 = int((y1 - left_b) / left_m)
                left_x2 = int((y2 - left_b) / left_m)

                if left_x1 < width / self.scale:
                    self.left_x1 = left_x1
                    self.left_line_state = 0
                if left_x2 < width / self.scale:
                    self.left_x2 = left_x2
                    self.left_line_state = 0
        else:
            self.left_line_state += 1

    def weightedFrame(self, img, alpha=0.8, beta=1., _lambda=0.):
        mask = np.zeros_like(img)
        width, height = self.getFrameSize(img)

        y1 = height
        y2 = int(height / 1.5)

        if self.left_x1 == None:
            self.left_x1 = 0
        if self.left_x2 == None:
            self.left_x2 = 0

        if self.right_x1 == None:
            self.right_x1 = width
        if self.right_x2 == None:
            self.right_x2 = width

        top_left_ROI = [self.left_x2, y2]
        top_right_ROI = [self.right_x2, y2]
        bottom_right_ROI = [self.right_x1, y1]
        bottom_left_ROI = [self.left_x1, y1]

        vertices = np.array(
            [[top_left_ROI, top_right_ROI, bottom_right_ROI, bottom_left_ROI]],
            dtype=np.int32)

        cv2.fillPoly(mask, vertices, [230, 100, 0])

        cv2.line(mask, (self.right_x1, y1), (self.right_x2, y2), [0, 255, 0], 5)
        cv2.line(mask, (self.left_x1, y1), (self.left_x2, y2), [0, 255, 0], 5)
        return cv2.addWeighted(img, alpha, mask, beta, _lambda)

    def bitwiseOrComposite(self, frame_A, frame_B):
        composite = cv2.bitwise_or(frame_A, frame_B)
        return composite

    def bitwiseAndComposite(self, frame_A, frame_B):
        composite = cv2.bitwise_and(frame_A, frame_B)
        return composite

    def createDisplayFrame(self, frame_title, frame):
        return cv2.imshow(frame_title, frame)

    def processVideoShowOutput(self):
        ret, frame = self.getOriginalFrame()
        width, height = self.getFrameSize(frame)

        if ret:

            img = self.scaleFrame(width / self.scale, height / self.scale, frame)
            lab = self.convertToLAB(img)
            channel = self.getColorChannel(lab, 2)

            left_edges = self.getEdges(channel)
            right_edges = self.getEdges(channel)

            self.calculateLinesRight(right_edges)
            self.calculateLinesLeft(left_edges)

            frame_with_lines = self.weightedFrame(img)

            self.createDisplayFrame("RightLines", self.getRightHough(right_edges)[1])
            self.createDisplayFrame("LeftLines", self.getLeftHough(left_edges)[1])
            self.createDisplayFrame("LAB Channel", channel)
            self.createDisplayFrame("weighted",frame_with_lines)

    def processLines(self):
        ret, frame = self.getOriginalFrame()
        width, height = self.getFrameSize(frame)

        if ret:
            img = self.scaleFrame(width / self.scale, height / self.scale, frame)

            lab = self.convertToLAB(img)

            channel = self.getColorChannel(lab, 2)

            left_edges = self.getEdges(channel)
            right_edges = self.getEdges(channel)

            self.calculateLinesRight(right_edges)
            self.calculateLinesLeft(left_edges)

    def setScale(self, scale):
        self.scale = scale

    def getCurrentLeftSpace(self):
        if self.left_x1 <= 0:
            return 100
        else:
            return int((float(self.left_x1) / (self.current_width / 2)) * 100)

    def getCurrentRightSpace(self):
        if self.right_x1 >= self.current_width:
            return 100
        else:
            return int((float(abs(240 - self.right_x1)) / (self.current_width / 2)) * 100)

    def move(self, left_space, right_space):
        if left_space - right_space > 2:
            if left_space < right_space:
                return "right"
            if right_space < left_space:
                return "left"
        else:
            return True

    def areRightLinesLost(self):
        if self.right_line_state > 10:

            return True
        else:
            return False

    def areLeftLinesLost(self):
        if self.left_line_state > 10:
            return True
        else:
            return False

'''

rl = Lines("drc3.mp4")

while True:

    rl.processVideoShowOutput()
    print rl.areLeftLinesLost(), rl.areRightLinesLost()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

rl.capture.release()
cv2.destroyAllWindows()
'''