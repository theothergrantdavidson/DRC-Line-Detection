rl = Lines("drc3.mp4")

while True:

    rl.processLines(True)
    print rl.areLeftLinesLost(), rl.areRightLinesLost()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

rl.capture.release()
cv2.destroyAllWindows()