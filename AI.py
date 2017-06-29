import time
import re
import serial
import cv2
import Lines
#import RPi.GPIO as GPIO
import sys
global leftMotor
global rightMotor
leftMotor = 0.0
rightMotor = 0.0
'''
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
right_motor=GPIO.PWM(17,100)
left_motor=GPIO.PWM(18,100)
right_motor.start(0)
left_motor.start(0)
'''







##ser = serial.Serial("/dev/ttyACM1", 115200, timeout=1)
##
### Regex for identifying distances and positions sent from the Arduino
##patternD = "Distance: ([0-9]*)"
##patternP = "Pos: ([0-9]*)"
##
### AI parameters
##collisionThreshold = 50
##
### Scan information
##distances = []
##positions = []



rl = Lines.Lines(0)

def cleanup():
        print("hello")
        '''
        right_motor.ChangeDutyCycle(0)
        left_motor.ChangeDutyCycle(0)
        ser.close()
        '''


##def recordTuple():
##        response = ser.readline().decode("utf-8", "ignore")
##
##        pos = re.search(patternP, response)
##        distance = re.search(patternD, response)
##
##        if(pos):
##                pos = int(pos.group(1))
##                positions.append(pos)
##
##        if(distance):
##                distance = int(distance.group(1))
##                distances.append(distance)
##
##def updateSections():
##        print ("Positions are: " + str(positions) + "\n")
##        print ("Distances are: " + str(distances) + "\n")
##
##        sections = [True, True, True]
##        for i in range(5, 115):
##                if(distances[i] < collisionThreshold and distances[i - 1] < collisionThreshold and distances[i + 1] < collisionThreshold):
##                        if(i < 40):
##                                sections[0] = False
##                        
##                        if(i >= 40 and i < 80):
##                                sections[1] = False
##                        
##                        if(i >= 80 and i <= 120):
##                                sections[2] = False
##
##        for i, s in enumerate(sections):
##                print ("Section " + str(i) + " is " + str(s))
                
        

try:
                                
        command = raw_input("w = go")
        if command=="w":

                while 1:

                        rl.processLines(True)
                        direction_value = rl.getDirection()
                        print direction_value
                        





                        
##                        # Get and record positions and distances
##                        recordTuple()
##
##                        # After a sweep is complete, update the sections with lidar information
##                        if(positions.length >= 120):
##                                updateSections()
##                        
##
##                        # Instructions to wheels
##                        if sections[1] == True or sections[1] == False:
##                                leftMotor = 100
##                                rightMotor = 95
##                                print("forward")
##
##                        #left turn
##
##                        elif sections[0] == True and sections[1] == False:
##                                leftMotor = 75
##                                rightMotor = 100
##                                print("left")
##
##                        # right turn
##                        elif sections[2] == True and sections[1] == False and sections[0] == False:
##                                leftMotor = 95
##                                rightMotor = 70
##                                print("right")
##
##                        positions.clear()
##                        distances.clear()
##                        sections.clear()
##
##                        right_motor.ChangeDutyCycle(rightMotor)
##                        left_motor.ChangeDutyCycle(leftMotor) 

                        if cv2.waitKey(30) & 0xFF == ord('q'):
                                break

                rl.capture.release()
                cv2.destroyAllWindows()

except KeyboardInterrupt:
        cleanup()

