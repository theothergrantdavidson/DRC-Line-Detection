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
        :ivar lost_line_threshold: A threshold where the program will believe that it can no longer see any lines
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
        self.lost_line_threshold = 10

    def getColorChannel(self, frame, channel):
        '''
        Method takes a 3-Channel frame and returns a specified color channel from that frame
        
        :param frame: A video frame
        :param channel: An integer from 0 - 2 which represents a color channel
        :return: The seperated color channel in the form of a 2-Channel frame
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
        :param low_threshold: Default value 50 - A value from 0 - 255 for the low threshold with which to choose edges
        :param high_threshold: Default value 100 - A value from 0 - 255 for the low threshold with which to choose edges
        :return: A 2-Channel black and white frame with detected edges
        '''
        return cv2.Canny(frame, low_threshold, high_threshold)

    def smoothFrame(self, frame, kernel_size=15):
        '''
        Uses a convulsion kernel to smooth an image based on an adaptive gaussian method, 
        the larger the kernel the longer the processing time
        
        :param frame: a 2-Channel or 3-Channel frame
        :param kernel_size: Must be an odd Integer
        :return: A frame that has had a gaussian blur applied to it
        '''
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def getRightHough(self, frame):
        '''
        This method calculates the hough lines of the right hand side of the frame and arranges them into an array, 
        it also returns the frame section which has been analysed. This is returned in the form of a tuple 
        (lines, masked_image)
        
        :param frame: A 2-Channel frame from getEdges() 
        :returns - lines: An array of lines which have been estimated from the input frame
        :returns - masked_image: A 2-Channel frame showing the edges that are being used in the estimation process
        '''
        rho = 1
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
        '''
        Takes an input frame and returns the width and height of that frame
        
        :param frame: A 2-Channel or 3-Channel frame
        :return: A tuple of integers of the width and height of the input frame (width, height)
        '''
        # defining a 3 channel or 1 channel frame
        if len(frame.shape) > 2:
            height, width = frame.shape[:2]
            return width, height
        else:
            height, width = frame.shape
            return width, height

    def getLeftHough(self, frame):
        '''
        This method calculates the hough lines of the left hand side of the frame and arranges them into an array, it also returns the frame section which has been analysed. This is returned in the form of a tuple (lines, masked_image)

        :param frame: A 2-Channel frame from getEdges()
        :returns - lines: An array of lines which have been estimated from the input frame
        :returns - masked_image: A 2-Channel frame showing the edges that are being used in the estimation process
        '''
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
        '''
        Runs liner regression on lines detected from the input frame to find a best fit coordinate for line marking
        on the right hand side
        
        :param frame: A 2-Channel frame from getEdges() method 
        '''
        lines = self.getRightHough(frame)[0] # Calculate lines in Hough space
        width, height = self.getFrameSize(frame) # Get size of input frame

        if lines is not None and len(lines) > 0:
            # If the hough algorithm produces lines, create empty arrays to the x coordinates and y coordinates
            right_lines_x = []
            right_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff) # get slope
                    slope_degree = abs(math.atan(slope)) * 100 # get slope degree

                    if slope_degree > 10 and slope > 0:
                        right_lines_x.append(x1)
                        right_lines_x.append(x2)

                        right_lines_y.append(y1)
                        right_lines_y.append(y2)

            y1 = height
            y2 = height / 1.5 # Top height of the Region of interest where lines are being calculated
            # Run linear regression to find x and y coordinates with best line fit
            if len(right_lines_y) > 0:
                right_m = np.polyfit(right_lines_x, right_lines_y, 1)[0]
                right_b = np.polyfit(right_lines_x, right_lines_y, 1)[1]
                # find slope intercepts
                right_x1 = int((y1 - right_b) / right_m)
                right_x2 = int((y2 - right_b) / right_m)
                # If either x1 or x2 is outside of their region of interest the results are ignored
                # reset empty line count when a line is found
                if right_x1 > width / self.scale:
                    self.right_x1 = right_x1
                    self.right_line_state = 0
                if right_x2 > width / self.scale:
                    self.right_x2 = right_x2
                    self.right_line_state = 0
        else:
            # Increment empty line count when there are no lines found
            self.right_line_state += 1

    def calculateLinesLeft(self, frame):
        '''
        Runs liner regression on lines detected from the input frame to find a best fit coordinate for line marking
        on the left hand side

        :param frame: A 2-Channel frame from getEdges() method 
        '''
        lines = self.getLeftHough(frame)[0] # Get estimated hough lines
        width, height = self.getFrameSize(frame) # Get size of input frame

        if lines is not None and len(lines) > 0:
            # If the hough algorithm produces lines, create empty arrays to the x coordinates and y coordinates
            left_lines_x = []
            left_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = np.float64(y2 - y1)
                x_diff = np.float64(x2 - x1)

                if x_diff > 0:
                    slope = np.divide(y_diff, x_diff) # Get slope
                    slope_degree = abs(math.atan(slope)) * 100 # Get slope degree

                    if slope_degree > 10 and slope < 0:
                        left_lines_x.append(x1)
                        left_lines_x.append(x2)
                        left_lines_y.append(y1)
                        left_lines_y.append(y2)

            y1 = height
            y2 = height / 1.5 # Top height of the Region of interest where lines are being calculated
            # Run linear regression to find x and y coordinates with best line fit
            if len(left_lines_y) > 0:
                left_m = np.polyfit(left_lines_x, left_lines_y, 1)[0]
                left_b = np.polyfit(left_lines_x, left_lines_y, 1)[1]
                # Find slope intercepts
                left_x1 = int((y1 - left_b) / left_m)
                left_x2 = int((y2 - left_b) / left_m)
                # If either x1 or x2 is outside of their region of interest the results are ignored
                # Reset empty line count when a line is found
                if left_x1 < width / self.scale:
                    self.left_x1 = left_x1
                    self.left_line_state = 0
                if left_x2 < width / self.scale:
                    self.left_x2 = left_x2
                    self.left_line_state = 0
        else:
            # Increment empty line count when there are no lines found
            self.left_line_state += 1

    def weightedFrame(self, img, alpha=0.8, beta=1., _lambda=0.):
        '''
        This function provides a visual representation of lines that are found for the right side and left side,
        this function is for debugging and not needed for the program to work.
        
        :param img: A 3-Channel input frame
        :param alpha: Default value 0.8 - Weight of the first array elements (input frame)
        :param beta: Default value 1.0 - Weight of the second array elements (lines that are overlaid)
        :param _lambda: Defalut value 0 - Scalar added to each sum of the array
        :return: A 3-Channel frame with lines overlaid where they have been found
        '''
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

    def createDisplayFrame(self, frame_title, frame):
        '''
        This method is for creating and returning a window to display 2-Channel or 3-Channel frames
        
        :param frame_title: A string which represents the title of the window that will be created
        :param frame: A 2-Channel or 3-Channel frame that will be rendered in the window
        :return: A Display window containing the input frame
        '''
        return cv2.imshow(frame_title, frame)

    def processLines(self, show_windows):
        '''
        This method is for processing the video input and filling the variables inside the constructor. This is the
        main program of the Lines module and should be run inside a while loop.
        
        :param show_windows: A boolean that for showing display windows for camera calibration and debugging
        '''
        ret, frame = self.getOriginalFrame()  # get return value and frame from video buffer
        width, height = self.getFrameSize(frame)  # get width and height of frame

        if ret:
            # scale image by the current image scale
            img = self.scaleFrame(width / self.scale, height / self.scale, frame)
            # convert img to LAB color space
            lab = self.convertToLAB(img)
            # Extract B channel from img
            channel = self.getColorChannel(lab, 2)
            # smooth channel with gaussin blur
            smooth = self.smoothFrame(channel, 3)
            # Get a canny edge frame from channel
            edges = self.getEdges(smooth)
            # Calculate the lines for left side and right side
            self.calculateLinesRight(edges)
            self.calculateLinesLeft(edges)

            if show_windows:
                # show video feedback for debugging
                frame_with_lines = self.weightedFrame(img)
                # Display right side line view
                self.createDisplayFrame("RightLines", self.getRightHough(edges)[1])
                # Display left side line view
                self.createDisplayFrame("LeftLines", self.getLeftHough(edges)[1])
                # Display current color channel that is being used for edges
                self.createDisplayFrame("LAB Channel", channel)
                # Display webcam input with lines overlaid
                self.createDisplayFrame("weighted",frame_with_lines)

    def setScale(self, scale):
        '''
        This method is for overriding the default scale value. The input is taken and divided by the original frame size
        of the input
        
        :param scale: Default value 2 - An integer or float to scale the size of the frame by
        '''
        self.scale = scale

    def getCurrentLeftSpace(self):
        '''
        A method which calculates the space between the center of the frame area and the bottom x position of the left
        line. Returns a percentage which represent how much space is left between the center of the frame and the
        left bottom x coordinate
        
        :return: A Integer which represents a percentage between 0 and 100
        '''
        if self.left_x1 <= 0:
            return 100
        else:
            return int((float(self.left_x1) / (self.current_width / 2)) * 100)

    def getCurrentRightSpace(self):
        '''
        A method which calculates the space between the center of the frame area and the bottom x position of the right
        line. Returns a percentage which represent how much space is left between the center of the frame and the
        right bottom x coordinate
        
        :return: A Integer which represents a percentage between 0 and 100 
        '''
        if self.right_x1 >= self.current_width:
            return 100
        else:
            return int((float(abs(240 - self.right_x1)) / (self.current_width / 2)) * 100)

    def areRightLinesLost(self):
        '''
        A method that checks how many times right lines have not been located
        
        :return: True if right lines can not be found, False otherwise
        '''
        if self.right_line_state > self.lost_line_threshold:
            return True
        else:
            return False

    def areLeftLinesLost(self):
        '''
        A method that checks how many times left lines have not been located

        :return: True if left lines can not be found, False otherwise
        '''
        if self.left_line_state > self.lost_line_threshold:
            return True
        else:
            return False

    def setLostLineThreshold(self, threshold):
        '''
        A method for overriding the value the lost line threshold which has a default value of 10
        
        :param threshold: An integer representing desired threshold to detect lost lines
        '''
        self.lost_line_threshold = threshold
