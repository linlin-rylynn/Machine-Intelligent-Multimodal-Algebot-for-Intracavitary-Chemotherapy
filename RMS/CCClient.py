#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import time
import xmlrpc.client
import socket
import os
#import ScriptDefine
#from Socket import SocketMng

#import modbus_tk.defines as cst
#import modbus_tk.modbus_tcp as modbus_tcp

#class MBMaster:
#    Modbus_Master = modbus_tcp.TcpMaster(host="127.0.0.1",port=10502)
#    def __init__(self):
#        self.Modbus_Master.set_timeout(5)



import struct

def ReadFloat(*args,reverse=False):
    for n,m in args:
        n,m = '%04x'%n,'%04x'%m
    if reverse:
        v = n + m
    else:
        v = m + n
    y_bytes = bytes.fromhex(v)
    y = struct.unpack('!f',y_bytes)[0]
    y = round(y,6)
    return y

def WriteFloat(value,reverse=False):
    print(WriteFloat)
    y_bytes = struct.pack('!f',value)
    print(y_bytes)
    y_hex = ''.join(['%02x' % i for i in y_bytes])
    print(y_hex)
    n,m = y_hex[:-4],y_hex[-4:]
    n,m = int(n,16),int(m,16)
    if reverse:
        v = [n,m]
    else:
        v = [m,n]
    return v

def ReadDint(*args,reverse=False):
    for n,m in args:
        n,m = '%04x'%n,'%04x'%m
    if reverse:
        v = n + m
    else:
        v = m + n
    y_bytes = bytes.fromhex(v)
    y = struct.unpack('!i',y_bytes)[0]
    return y

def WriteDint(value,reverse=False):
    y_bytes = struct.pack('!i',value)
    # y_hex = bytes.hex(y_bytes)
    y_hex = ''.join(['%02x' % i for i in y_bytes])
    n,m = y_hex[:-4],y_hex[-4:]
    n,m = int(n,16),int(m,16)
    if reverse:
        v = [n,m]
    else:
        v = [m,n]
    return v
#################################################################################################
class CCClient(object):
    clientIP = '0.0.0.0'
    clientPort = 10003
    xmlrpcAddr='http://127.0.0.1:20000'
    params=[]
    
    def connectTCPSocket(self,IP):
        clientIP=IP
        xmlrpcAddr='http://'
        xmlrpcAddr+=clientIP
        xmlrpcAddr+=':20000'
        print(xmlrpcAddr)
        self.rpcClient = xmlrpc.client.ServerProxy(self.xmlrpcAddr)
        return self.tcp.connect((clientIP, self.clientPort))
        
    #moveC
    def waitMoveDone(self):
        time.sleep(0.02)
        nDisableCNT=0
        while True:
            if (nDisableCNT>=5):
                time.sleep(0.01)
                os._exit(0)
            ## check need loop first
            ## only normal module, 
            ## because debug module only pause by breakpoint. 
            #self.is_need_loop_in_waitMoveDone()
            
            # read CC stauts
            ret = self.readRobotState()
            
            
            # [ReadRobotState,OK, movingState-2, EnableState-3, errorState-4,
            # errorCode-5, errorAxis-6, Breaking-7, Pause-8, 
            # emergency-9, SafeGraud-10, Electfify-11, sysboradConnect-12, blendingDone-13]
            
            # disable or error or emergency or BlackOut or sysBorad disconnect
            # in this case, CC will be send Stop cmd to py in anther threading
            # so it just need waiting in here    
            if (ret[3] == '0'):
                nDisableCNT+=1
                #self.sendHRLog(2,'[script]EnableState['+ret[3]+'],count['+str(nDisableCNT)+'] error')
                continue
            else:
                nDisableCNT=0
                   
            if (ret[4] == '1' or ret[9] == '1' or ret[11] == '0' or ret[12] == '0' ):
                log = ('[script]errorState['+ret[4]+'],emergency['+ret[9]+'],Electfify['+ret[11]+']')
                #self.sendHRLog(2,str(log))
                time.sleep(0.1)
                os._exit(0)

            # SafeGraud
            # if it is stop , CC will be send Stop cmd in anther threading
            # if it is puase, CC will be send Pause in anther threading, and it will pause in "is_need_loop_here()"
            # so it just need continue in here;            
            elif ret[10] == '1':

                time.sleep(0.01)
                continue

            # pause
            # also continue 
            elif ret[8] == '1':

                time.sleep(0.01)
                continue

            # moveing
            # continue for next time
            elif ret[2] == '1':

                time.sleep(0.01)
                continue

            # moveing over
            elif ret[2] == '0':
                #print ("move over")
                break

            # unknow status  maybe exit is more safe                
            else: 
                log = ('[script]waitMoveDone unknow status exit')
                os._exit(0)
        return
    #movej movel
    def moveJ(self, joint):
        command = 'MoveJ,0,'
        for i in range(0, 6):
            command += str(joint[i]) + ','
        command += ';'
        return self.sendAndRecv(command)
    def moveL(self, pose):
        command = 'MoveL,0,'
        for i in range(0, 6):
            command += str(pose[i]) + ','
        command += ';'
        return self.sendAndRecv(command)
    def waitBlendingDone(self):
        time.sleep(0.02)
        nDisableCNT=0
        while True:
            if (nDisableCNT>=5):
                time.sleep(0.01)
                os._exit(0)
            ## check need loop first
            ## only normal module, 
            ## because debug module only pause by breakpoint. 
            #self.is_need_loop_in_waitMoveDone()
            
            # read CC stauts
            ret = self.readRobotState()
            #print ret
            
            # [ReadRobotState,OK, movingState-2, EnableState-3, errorState-4, errorCode-5, errorAxis-6, Breaking-7, Pause-8, emergency-9, SafeGraud-10, Electfify-11, sysboradConnect-12 blendingDone-13]
            #
            # disable or error or emergency or BlackOut or sysBorad disconnect
            # in this case, CC will be send Stop cmd to py in anther threading
            # so it just need waiting in here
            if (ret[3] == '0'):
                nDisableCNT+=1
                #self.sendHRLog(2,'[script]EnableState['+ret[3]+'],count['+str(nDisableCNT)+'] error')
                continue
            else:
                nDisableCNT=0
            
            if (ret[4] == '1' or ret[9] == '1' or ret[11] == '0' or ret[12] == '0' ):
                log = ('[script]errorState['+ret[4]+'],emergency['+ret[9]+'],Electfify['+ret[11]+']')
                #self.sendHRLog(2,str(log))
                time.sleep(0.1)
                os._exit(0)
            # SafeGraud
            # if it is stop , CC will be send Stop cmd in anther threading
            # if it is puase, CC will be send Pause in anther threading, and this thread will pause in "is_need_loop_here()"
            # so it just need continue in here;            
            elif ret[10] == '1':
                #print 'ret[10]=='+ret[10]
                time.sleep(0.01)
                continue

            # pause
            # also continue 
            elif ret[8] == '1':
                #print 'ret[8]=='+ret[8]
                time.sleep(0.01)
                continue

            # blending over
            elif ret[13] == '1':
                #log = ('[script]ret[13]=='+ret[13])
                #self.sendHRLog(3,str(log))
                break

            # blending 
            elif ret[13] == '0':
                #print 'ret[13]=='+ret[13]
                time.sleep(0.01)
                continue


            # unknow status  maybe exit is more safe                
            else: 
                log = ('[script]waitBlendingDone unknow status exit')
                #self.sendHRLog(3,str(log))
                os._exit(0)

        return
        
    # set speed percent
    # vel: double type
    # vel>0 && vel < 1
    #
    # @output if it is work OK ,
    #         it will retrun "SetOverride,OK,;" ,
    #         
    #         if it work error, it will ret "SetOverride,FAIL,20011,;", 20011 is a errorcode example

    def SetOverride(self, vel):
        command = 'SetOverride,0,' + str(vel) + ',;'
        return self.sendAndRecv(command)
    
    def SetPayload(self, mass,Center_X,Center_Y,Center_Z):
        command = 'SetPayload,0,'
        command += str(mass)
        command += ','
        command += str(Center_X)
        command += ','
        command += str(Center_Y)
        command += ','
        command += str(Center_Z)
        command += ',;'
        return self.sendAndRecv(command)

    # setTCP
    def setTCP(self, TCP):
        command = 'SetTCPByName,0,'
        command += str(TCP) + ',;'
        return self.sendAndRecv(command)


    # set UCS
    def setUCS(self, UCS):
        command = 'SetUCSByName,0,'
        command += str(UCS) + ',;'
        return self.sendAndRecv(command)  
    
    # setTCP
    def ReadTCP(self, TCP):
        command = 'ReadTCPByName,0,'
        command += str(TCP) + ',;'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    def ConfigTCP(self,name,TCP):
        command = 'ConfigTCP,'
        command += str(name)
        command += ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        command += ';'
        return self.sendAndRecv(command)  

    # set UCS
    def ReadUCS(self, UCS):
        command = 'ReadUCSByName,0,'
        command += str(UCS) + ',;'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    def ConfigUCS(self,name,UCS):
        command = 'ConfigUCS,'
        command += str(name)
        command += ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        return self.sendAndRecv(command)

    #SetMaxPcsRange
    def SetMaxPcsRange(self, pMax, pMin,pUcs):
        command = 'SetMaxPcsRange,0,'
        command += str(pMax[0])
        command += ','
        command += str(pMax[1])
        command += ','
        command += str(pMax[2])
        command += ','
        command += str(180)
        command += ','
        command += str(180)
        command += ','
        command += str(180)
        command += ','
        command += str(pMin[0])
        command += ','
        command += str(pMin[1])
        command += ','
        command += str(pMin[2])
        command += ','
        command += str(-180)
        command += ','
        command += str(-180)
        command += ','
        command += str(-180)
        command += ','
        for i in range(0, 6):
            command += str(pUcs[i]) + ','
        command += ';'
        return self.sendAndRecv(command)

    #UcsTcp2Base
    def UcsTcp2Base(self, UcsTcp, TCP, UCS):
        command = 'UcsTcp2Base,0,'
        for i in range(0, 6):
            command += str(UcsTcp[i]) + ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    #Base2UcsTcp
    def Base2UcsTcp(self, Base, TCP, UCS):
        command = 'Base2UcsTcp,0,'
        for i in range(0, 6):
            command += str(Base[i]) + ','
        for i in range(0, 6):
            command += str(TCP[i]) + ','
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    #PCS2ACS
    def PCS2ACS(self, rawPCS, rawACS, tcp, ucs):
        command = 'PCS2ACS,0,'
        for i in range(0, 6):
            command += str(rawPCS[i]) + ','
        for i in range(0, 6):
            command += str(rawACS[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    def startServo(self, servoTime, lookaheadTime):
        command = 'StartServo,0,'
        command += str(servoTime) + ',' + str(lookaheadTime) + ',;'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal

    def pushServoP(self, pose):
        command = 'PushServoP,0,'
        for i in range(0, 18):
            command += str(pose[i]) + ','
        command += ';'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal

    def pushServoJ(self, pose):
        command = 'PushServoJ,0,'
        for i in range(0, 6):
            command += str(pose[i]) + ','
        command += ';'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal

    def MovePath(self, trajectName):
        command = 'MovePath,0,'
        command += str(trajectName) + ','
        command += ';'
        return self.sendAndRecv(command)

    #moveType 
    #"MoveJ" 0   "MoveL" 1   "MoveC" 2
    #relMoveType 0 绝对值, 1 相对值
    def WayPointRel(self, type, usePointList, Point_PCS, Point_ACS, relMoveType, nAxisMask, Axis0,Axis1,Axis2,Axis3,Axis4,Axis5, tcp, ucs, speed, Acc, radius,isJoint, isSeek, bit, state, cmdID):
        command = 'WayPointRel,0,'
        command += str(type) + ','
        command += str(usePointList) + ','
        for i in range(0, 6):
            command += str(Point_PCS[i]) + ','
        for i in range(0, 6):
            command += str(Point_ACS[i]) + ','
        command += str(relMoveType) + ','
        for i in range(0, 6):
            command += str(nAxisMask[i]) + ','
        command += str(Axis0) + ','
        command += str(Axis1) + ','
        command += str(Axis2) + ','
        command += str(Axis3) + ','
        command += str(Axis4) + ','
        command += str(Axis5) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.sendAndRecv(command)

    def WayPointEx(self, type, points, RawACSpoints, tcpname, ucs, speed, Acc, radius,isJoint, isSeek, bit, state, cmdID):
        tcp=self.ReadTCP(tcpname)
        command = 'WayPointEx,0,'
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        for i in range(0, 6):
            command += str(ucs[i]) + ','
        for i in range(0, 6):
            command += str(tcp[i]) + ','
        #command += str(tcp) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(type) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.sendAndRecv(command)

    def WayPoint(self, type, points, RawACSpoints, tcp, ucs, speed, Acc, radius,isJoint, isSeek, bit, state, cmdID):
        command = 'WayPoint,0,'
        for i in range(0, 6):
            command += str(points[i]) + ','
        for i in range(0, 6):
            command += str(RawACSpoints[i]) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(type) + ','
        command += str(isJoint) + ','
        command += str(isSeek) + ','
        command += str(bit) + ','
        command += str(state) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.sendAndRecv(command)

    def MoveC(self, StartPoint, AuxPoint, EndPoint, fixedPosure, nMoveCType, nRadLen,speed, Acc, radius, tcp, ucs, cmdID):
        command = 'MoveC,0,'
        for i in range(0, 6):
            command += str(StartPoint[i]) + ','
        for i in range(0, 6):
            command += str(AuxPoint[i]) + ','
        for i in range(0, 6):
            command += str(EndPoint[i]) + ','
        command += str(fixedPosure) + ','
        command += str(nMoveCType) + ','
        command += str(nRadLen) + ','
        command += str(speed) + ','
        command += str(Acc) + ','
        command += str(radius) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.sendAndRecv(command)

    def MoveZ(self, StartPoint, EndPoint, PlanePoint, Speed, Acc, WIdth, Density, EnableDensity, EnablePlane, EnableWaiTime, PosiTime, NegaTime, Radius, tcp, ucs, cmdID):
        command = 'MoveZ,0,'
        for i in range(0, 6):
            command += str(StartPoint[i]) + ','
        for i in range(0, 6):
            command += str(EndPoint[i]) + ','
        for i in range(0, 6):
            command += str(PlanePoint[i]) + ','
        command += str(Speed) + ','
        command += str(Acc) + ','
        command += str(WIdth) + ','
        command += str(Density) + ','
        command += str(EnableDensity) + ','
        command += str(EnablePlane) + ','
        command += str(EnableWaiTime) + ','
        command += str(PosiTime) + ','
        command += str(NegaTime) + ','
        command += str(Radius) + ','
        command += str(tcp) + ','
        command += str(ucs) + ','
        command += str(cmdID) + ','
        command += ';'
        return self.sendAndRecv(command)

    #SetForceControlState
    def SetForceControlState(self, state):
        command = 'SetForceControlState,0,' + str(state) + ',;'
        retData = self.sendAndRecv(command)
        if retData[1] == 'OK':
            while True:
                command = 'ReadFTControlState,0,;'
                retData = self.sendAndRecv(command)
                if state == 1:
                    if int(retData[2]) == 2:
                        break
                elif state == 0:
                    if int(retData[2]) != 2:
                        break
                time.sleep(0.01)
        return True


    #Pose_ReadJoint
    def Pose_ReadJoint(self):
        retData = self.readActualPos()
        #print retData
        retData.pop()
        retData.pop()
        retData.pop()
        retData.pop()
        retData.pop()
        retData.pop()
        #retData = list(map(int,retData))
        #print (retData)
        #print (retData[0])
        return retData
		
    #Pose_ReadPos
    def Pose_ReadPos(self):
        retData = self.readActualPos()
        #print retData
        del retData[0]
        del retData[0]
        del retData[0]
        del retData[0]
        del retData[0]
        del retData[0]
        return retData
    	
    #Pose_Add
    def Pose_Add(self, pos1, pos2):
        command = 'Pose_Add,0,'
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData
		
	#Pose_Sub
    def Pose_Sub(self, pos1, pos2):
        command = 'Pose_Sub,0,'
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData
    
	#Pose_Trans
    def Pose_Trans(self, pos1, pos2):
        command = 'PoseTrans,0,'
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        for i in range(0, 6):
            command += str(pos2[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    #Pose_Trans
    def Pose_Inverse(self, pos1):
        command = 'PoseInverse,0,'
        for i in range(0, 6):
            command += str(pos1[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    #Pose_DefdFrame
    def Pose_DefdFrame(self, UCS, pos1, pos2, pos3, pos4, pos5, pos6):
        command = 'DefdFrame,0,'
        for i in range(0, 6):
            command += str(UCS[i]) + ','
        for i in range(0, 3):
            command += str(pos1[i]) + ','
        for i in range(0, 3):
            command += str(pos2[i]) + ','
        for i in range(0, 3):
            command += str(pos3[i]) + ','
        for i in range(0, 3):
            command += str(pos4[i]) + ','
        for i in range(0, 3):
            command += str(pos5[i]) + ','
        for i in range(0, 3):
            command += str(pos6[i]) + ','
        command += ';'
        retData = self.sendAndRecv(command)
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    
    # read Robot state
    def readRobotState(self):
        return self.sendAndRecv('ReadRobotState,0,;')
        

    # stop moveing
    # no parameter
    def stop(self):
        return self.sendAndRecv('GrpStop,0,;')


    # ReadActualPos
    # 
    # @output = [J1,J2,J3,J4,J5,J6,X,Y,Z,A,B,C]
    # !! the points in Cartesian coordinates is based on the base coordinates (not user coordinates)
    #
    def readActualPos(self):
        retData = self.sendAndRecv('ReadActPos,0,;')
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData
    
    def ReadActPos(self):
        retData = self.sendAndRecv('ReadActPos,0,;')
        #print retData
        del retData[0]
        del retData[0]
        retData.pop()
        return retData

    # sendCmdID
    # No output 
    def sendCmdID(self, CmdID,ThreadID):
        self.rpcClient.SetCurCmdID(str(CmdID),str(ThreadID))
        #command = 'SendCmdID,0,' + str(CmdID) + ',' + str(ThreadID)  + ',;'
        #retData = self.sendAndRecv(command)
        #return retData
    
    # send the finish code 
    # if py has running finish 
    # py should send a finish message to CC
    # No output 
    def sendScriptFinish(self,errorcode):
        command = 'SendScriptFinish,0,'+ str(errorcode) + ',;'
        self.tcp.send(command.encode())
        self.tcp.recv(self.clientPort).decode()
        

    def sendScriptError(self,msg):
        self.rpcClient.SendScriptError(str(msg),str(""))
    
    def sendHRLog(self,nLevel,msg):
        self.rpcClient.HRLog(int(nLevel),str(msg))

    # sendVarValue
    # No output 
    def sendVarValue(self, VarName, Value):
        if isinstance(Value,list):
            ValueStr='['
            for i in range(0, 6):
                ValueStr += str(Value[i])
                if i!=5:
                    ValueStr += ','

            ValueStr += ']'
            #command = 'SendVarValue,0,' + str(VarName) + ','
            #for i in range(0, 6):
            #    command += str(Value[i]) + ','
            #command += ';'
        else:
            ValueStr=str(Value)
        #gVar = globals()
        #gVar[VarName]=Value
        self.rpcClient.SendVarValue(str(VarName),ValueStr)
        #return retData  

    # config IO
    def readCI(self, bit):
        command = 'ReadBoxCI,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def readCO(self, bit):
        command = 'ReadBoxCO,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def setCO(self, bit, state):
        command = 'SetBoxCO,' + str(bit) + ',' + str(state) + ',;'
        return self.sendAndRecv(command)
    
    def cdsSetIO(self, nEndDOMask,nEndDOVal,nBoxDOMask,nBoxDOVal,nBoxCOMask,nBoxCOVal,nBoxAOCH0_Mask,nBoxAOCH0_Mode,nBoxAOCH1_Mask,nBoxAOCH1_Mode,dbBoxAOCH0_Val,dbBoxAOCH1_Val):
        command = 'cdsSetIO,'
        command += str(nEndDOMask) + ','
        command += str(nEndDOVal) + ','
        command += str(nBoxDOMask) + ','
        command += str(nBoxDOVal) + ','
        command += str(nBoxCOMask) + ','
        command += str(nBoxCOVal) + ','
        command += str(nBoxAOCH0_Mask) + ','
        command += str(nBoxAOCH0_Mode) + ','
        command += str(nBoxAOCH1_Mask) + ','
        command += str(nBoxAOCH1_Mode) + ','
        command += str(dbBoxAOCH0_Val) + ','
        command += str(dbBoxAOCH1_Val) + ','
        command += ';'
        return self.sendAndRecv(command)
    
    def SetTrackingState(self, state):
        command = 'SetTrackingState,0,' + str(state) + ',;'
        return self.sendAndRecv(command)

    def HRApp(self,name,param):
        command = 'HRAppCmd,'
        command += str(name) + ','
        command += str(param) + ','
        command += ';'
        return self.sendAndRecv(command)


    #  
    def readDO(self, bit):
        command = 'ReadBoxDO,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def readDI(self, bit):
        command = 'ReadBoxDI,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def setDO(self, bit, state):
        command = 'SetBoxDO,' + str(bit) + ',' + str(state) + ',;'
        return self.sendAndRecv(command)
    
    def readBoxAI(self, bit):
        command = 'ReadBoxAI,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return float(retData[2])

    def readAO(self, bit):
        command = 'ReadBoxAO,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        #return int(retData[2])
        return float(retData[3])

    def SetBoxAO(self,index,value,pattern):
        command = 'SetBoxAO,' + str(index) + ',' + str(value) + ',' + str(pattern) + ',;'
        return self.sendAndRecv(command)

    #  end io
    def readEI(self, bit):
        command = 'ReadEI,0,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def readEO(self, bit):
        command = 'ReadEO,0,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])

    def setEO(self, bit, state):
        command = 'SetEndDO,0,' + str(bit) + ',' + str(state) + ',;'
        return self.sendAndRecv(command)

    def readEAI(self, bit):
        command = 'ReadEAI,0,' + str(bit) + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])
    
    def InitMovePathL(self,strCmd,vel,acc,jerk,ucs,tcp):
        command = 'InitMovePathL,0,'
        command += strCmd + ','
        command += str(vel) + ','
        command += str(acc) + ','
        command += str(jerk) + ','
        command += ucs + ','
        command += tcp + ','
        command += ';'
        print(command)
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal
    
    def PushMovePathL(self,strCmd,pos):
        command = 'PushMovePathL,0,'
        command += strCmd
        command += ','
        for i in range(0, 6):
            command += str(pos[i]) + ','
        command += ';'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal
    
    def PushMovePaths(self,strName,nMoveType,pointsCNT,points):
        command = 'PushMovePaths,0,'
        command += strName
        command += ','
        command += str(nMoveType)
        command += ','
        command += str(pointsCNT)
        command += ','
        for pos in points:
           command += str(pos) + ','
        
        command += ';'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal
        
    def MovePathL(self,strCmd):
        command = 'MovePathL,0,'
        command += strCmd
        command += ',;'
        self.tcp.send(command.encode())
        retSignal = self.tcp.recv(self.clientPort)
        return retSignal
    
    # modbus 
    def setModbus(self,deviceName,varName,value):
        command = 'SetExDeviceData,' + deviceName + ',' + varName + ',' + str(value) + ',;'
        return self.sendAndRecv(command)

    def getModbus(self,deviceName,varName):
        command = 'ReadExDeviceData,' + deviceName + ',' + varName + ',;'
        retData = self.sendAndRecv(command)
        return int(retData[2])
    
    def sendAndRecv(self, cmd):
        self.tcp.send(cmd.encode())
        #print cmd
        ret = self.tcp.recv(self.clientPort).decode()
        retData = ret.split(',')
        #print(retData)
        if len(retData) < 3:
            #self.sendHRLog(2,'[script]sendAndRecv exit with ServerReturnError')
            #self.sendScriptFinish(ScriptDefine.ErCode.ServerReturnError)
            #self.closeTCPSocket()
            os._exit(0)

        if retData[0] == "errorcmd":
            #self.sendHRLog(2,'[script]sendAndRecv exit with errorcmd')
            #self.sendScriptFinish(ScriptDefine.ErCode.CientCmdError)
            #self.closeTCPSocket()
            os._exit(0)

        if retData[1] == "Fail":
            #self.sendHRLog(2,'[script]sendAndRecv exit with Fail['+retData[2]+']')
            #self.sendScriptFinish(retData[2])
            #self.closeTCPSocket()
            os._exit(0)
        
        return retData

    def closeTCPSocket(self):
        return self.tcp.close()

    def __init__(self):
        #self.rpcClient = xmlrpc.client.ServerProxy(self.xmlrpcAddr)
        self.tcp = socket.socket()
        #self.socket = SocketMng()
        return

    
