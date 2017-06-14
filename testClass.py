import logging
logging.basicConfig(filename='RoadLineDetection.log',level=logging.DEBUG)

try:
    import math
    import cv2
    import numpy as np
except ImportError, e:
    logging.info(e.message)

class RoadLineDetection():

    def __init__(self, capture):
        self.capture = cv2.VideoCapture(capture)
        ret, frame = self.capture.read()
        h, w = frame.shape[:2]
        self.current_height = h
        self.current_width = w

    '''
    This method takes a cv2 frame and resizes it.
    Parameters:
        width - a percentage of the original size of the input frame width
        height - a percentage of the original size of the input frame height
    Returns:
        Resized frame according to the input percentatges
    '''
    def resizeFrame(self, frame, width, height):
        h, w = frame.shape[:2]
        returnFrame = cv2.resize(frame,((int(w * width)), (int(h * height))), interpolation=cv2.INTER_CUBIC)
        new_h, new_w = returnFrame.shape[:2]
        self.current_width = new_w
        self.current_height = new_h
        return returnFrame
    '''
    This method creates a numpy array that is used for a region of interest(ROI) in the
    getRegionOfInterest() method, the ROI is a 4 point region of interest.
    Parameters:
        top_left - and array of 2 elements defining the top left hand corner of the ROI
        top_right - and array of 2 elements defining the top right hand corner of the ROI
        bottom_left - and array of 2 elements defining the bottom left hand corner of the ROI
        right_right - and array of 2 elements defining the bottom left hand corner of the ROI
    Returns:
        vertices - numpy array in the shape of a rectangle
    '''
    def regionOfInterestVerts(self, top_left, top_right, bottom_right, bottom_left):
        vertices = np.array([[top_left, top_right, bottom_left, bottom_right]],dtype=np.int32)
        return vertices

    '''
    This method takes a cv2 frame and converts from BGR to HSV color space
    Parameters:
        frame - cv2 video frame
    Returns:
        frame converted to HSV color space
    '''
    def convertToHSV(self, frame):
        return cv2.cvtColor(frame ,cv2.COLOR_BGR2HSV)

    '''
    This method takes a cv2 frame and converts from BGR to HLS color space
    Parameters:
        frame - cv2 video frame
    Returns:
        frame converted to HLS color space
    '''
    def convertToHLS(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    '''
    This method takes a cv2 frame and converts from BGR to Grey color space
    Parameters:
        frame - cv2 video frame
    Returns:
        frame converted to grey color space
    '''
    def convertToGrey(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    This method takes a cv2 frame and converts from BGR to XYZ color space
    Parameters:
        frame - cv2 video frame
    Returns:
        frame converted to XYZ color space
    '''
    def convertToXYZ(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)

    '''
    This method takes a cv2 frame and converts from BGR to LAB color space
    Parameters:
        frame - cv2 video frame
    Returns:
        frame converted to LAB color space
    '''
    def convertToLAB(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    '''
    This method takes a cv2 frame instance and returns a version of that frame with a masked
    region of interest  
    Parameters:
        img - cv2 video frame
        vertices - return value of regionOfInterestVerts method
    Returns:
        masked_image - A video frame that has been masked with a region of interest
    '''
    def getRegionOfInterest(self, frame):

        vertices = self.regionOfInterestVerts([0, self.current_height / 2],
                                                    [self.current_width, self.current_height / 2],
                                                    [0, self.current_height],
                                                    [self.current_width, self.current_height])
        # create an empty array in the same structure as the img array except with zero for every value
        mask = np.zeros_like(frame)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = frame.shape  # count of how many channels there are
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # fill pixels in mask polygon
        cv2.fillPoly(mask, vertices, 255)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(frame, mask)
        return masked_image

    '''
    This method reads the capture source and returns a frame, will keep reading a video source if placed in
    a while loop
    Parameters:
        null
    Returns:
        frame from capture source
    '''
    def getCaptureFrame(self):
        ret, frame = self.capture.read()
        return frame

    '''
    This method returns true if the capture source is still readable, will keep reading a video source if placed in
    a while loop
    Parameters:
        null
    Returns:
        Boolean - Based on whether the video capture source is still readable
    '''
    def getCaptureState(self):
        ret, frame = self.capture.read()
        return ret

    '''
    This method releases the capture source
    Parameters:
        null
     Returns:
        null
    '''
    def releaseCaptureSourse(self):
        self.capture.release()

    '''
    This runs a canny edge detection algorithm over the input frame
    Parameters:
        frame - cv2 video frame
        low_threshold - low threshold cut off (integer)
        high_threshold - high threshold cut off (integer)
    Returns:
        frame - returns a frame that has has a canny edge detection run over it
    '''
    def canny(self, frame, low_threshold=10, high_threshold=150):
        return cv2.Canny(frame, low_threshold, high_threshold)

    '''
    This method calcualtes the x1, y1, x2, y2 vectors of lines that are found in the input
    image
    Parameters:
        frame - frame that has been transformed by canny()
    Returns:
        lines - an array of line values found on the image
    '''
    def houghLines(self, frame):
        # Hough Transform
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 0  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        return lines

    '''
    This method calcualtes the x1, y1, x2, y2 vectors of lines that are found in the input
    image
    Parameters:
        frame - cv2 image frame
        channel - a number from 0 - 2 represething the color desired color channel
    Returns:
       frame - a grey image of the desired color channel
    '''
    def getColorChannel(self, frame, channel):
        return frame[:, :, channel]

    '''
    This method calcualtes the x1, y1, x2, y2 vectors of lines that are found in the input
    image
    Parameters:
        frame - cv2 image frame
        channel - a number from 0 - 2 represething the color desired color channel
    Returns:
        frame - a grey image of the desired color channel
    '''
    def sortLinesLeftRight(self, lines):
        if lines is not None:
            right_lines_x = []
            right_lines_y = []

            left_lines_x = []
            left_lines_y = []

            for x1, y1, x2, y2 in lines[0]:
                y_diff = y2 - y1
                x_diff = x2 - x1
                if x_diff != 0:

                    slope = np.divide(y_diff, x_diff)
                    slope_degree = abs(math.atan(np.divide(y_diff, x_diff))) * 100

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

                return right_lines_x, right_lines_y, left_lines_x, left_lines_y

det = RoadLineDetection("test3.mp4")

while True:
    if det.getCaptureState() == True:
        img = det.getCaptureFrame()
        resized_img = det.resizeFrame(img, 0.5, 0.5)
        channel = det.getColorChannel(det.convertToLAB(resized_img), 2)
        canny = det.canny(channel)
        reg = det.getRegionOfInterest(canny)
        cv2.imshow("thing",det.canny(channel))
        lines = det.houghLines(canny)
        print canny.shape
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

det.releaseCaptureSourse()
cv2.destroyAllWindows()