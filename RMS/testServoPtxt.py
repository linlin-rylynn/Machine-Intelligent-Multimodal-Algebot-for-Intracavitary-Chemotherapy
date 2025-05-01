#
"This file is used to test the robotic arm motion through a predefined path"
#

from RMS.Robotcient import CCClient
import time
import numpy as np

# Read position data file
fname = "servoPdata.txt"

# Read each line of data into the teach array
with open(fname, 'r+', encoding='utf-8') as f:
    teach = [i[:-1].split(',') for i in f.readlines()]

# Connect to robot
cps = CCClient()    # Construct robot library class
cps.connectTCPSocket('192.168.0.10')   # Connect to robot IP address
cps.SetOverride(0.010)   # Set speed ratio to 10% of maximum end effector speed constraint

# Move to first position
joints = teach[0][0:6]
print(cps.moveL(joints))
print(joints)

# Wait for movement completion
cps.waitMoveDone()

start = time.perf_counter()
servoTime = 0.025   # Set servo cycle to 25ms, recommended minimum not less than 15ms
# Set lookahead time, larger lookahead time makes trajectory smoother but more laggy
lookaheadTime = 0.2 # Set lookahead time to 200ms, recommended between 0.05s~0.2s
# Enable online control (ServoP)
print(cps.startServo(servoTime, lookaheadTime))  # start Servo
i = 0
nodeTime = start
failTime = 0
loopTime = 1    # How many times to run the loop
flag = True # Stop flag

# Call pushServoP interface for online servo control of the robot
while(flag):
    ii = 0
    while flag:
        if i >= len(teach) * loopTime:
            flag = False    # Run loopTime times, then stop
            break
        teach[ii][6:18] = ['0','0','0','0','0','0','0','0','0','0','0','0'] # Add UCS,TCP coordinates

        #teach[ii][6:18] = ['25.62','431.452','108.987','-179.993','-1.988','112.232','0','0','0','0','0','0']
        recv = (cps.pushServoP(teach[ii][0:18])).decode()  # update servo position
        recv += "  " + 'Fail Times : ' + str(failTime)
        print(recv)
        recvData = recv.split(',')
        if (recvData[1] != 'OK'):
            failTime = failTime + 1 # Count how many times failed

        # Ensure position is sent according to servo cycle time
        while True:  # waiting for next servo time
            currentTime = time.perf_counter()
            if currentTime - nodeTime > servoTime:
                nodeTime = currentTime
                break
            time.sleep(0.0001)
        i += 1
        ii += 1
    end = time.perf_counter()
    print(end - start)  # Print how much time elapsed
    print(len(teach))
    loopTime = loopTime + 1
