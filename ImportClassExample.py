import cv2
import Lines

rl = Lines.Lines()

while True:
    # processLines is the method which calculates the lines
    # Boolean is True to display video out put windows for calibration, False to turn them off
    rl.processLines(True)

    # This needs to be called after processLine and gives a direction 0 1 -1
    direction_value = rl.getDirection()
    # This needs to be in the main while loop with processLines and getDirection can be put down the bottom of the loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
# These bottom two lines can be thrown dowon the bottom of the program outside of the loop
# cv2 gets pissy if you don't
rl.capture.release()
cv2.destroyAllWindows()