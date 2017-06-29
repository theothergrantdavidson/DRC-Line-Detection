import cv2
import Lines


rl = Lines.Lines("drc1.mp4")


while True:

    rl.processLines(True)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

rl.capture.release()
cv2.destroyAllWindows()