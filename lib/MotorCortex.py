import lib.globals as brain
import serial
import time

class MotorCortex:
    memory = None
    def __init__(self, port):
        print ("**** MotorCortex: Initializing motor cortex")
        #StepperControllerInterface - COM port, baud rate, xSpeed, ySpeed, xAcceleration, yAcceleration, xHomeSpeed, yHomeSpeed, xHomeAcceleration, yHomeAcceleration, xHomeOffset, yHomeOffset, xmax, ymax
        self.steppers = StepperControllerInterface(port, 250000, 6000, 10000, 2000, 4000, 200, 200, 20000, 20000, -700, -350, 1400, 1800)
        print ("=> MotorCortex is initialized")
        self.steppers.Home()
        self.steppers.StepperY.Move(300)

        time.sleep(15)
        pass

    def update(self):
        if brain.spine.visualCortex.currentlyTracking is not None:
            trackedPerson = brain.spine.visualCortex.currentlyTracking
            if trackedPerson.xOffset > 0.5:
                self.steppers.StepperX.Move(((trackedPerson.xOffset/2) * -1) * 1)
            elif trackedPerson.xOffset < -0.5:
                self.steppers.StepperX.Move(((trackedPerson.xOffset/2) * -1) * 1)
            if trackedPerson.yOffset > 0.5:
                self.steppers.StepperY.Move(((trackedPerson.yOffset/2) * -1) * 1)
            elif trackedPerson.yOffset < -0.5:
                self.steppers.StepperY.Move(((trackedPerson.yOffset/2) * -1) * 1)
        pass


class StepperMotor:
    def __init__(self, _serial, _axis):
        self.ser = _serial
        self.axis = _axis

    def MoveTo(self, _position):
        command = 'MOVE' + self.axis + 'TO,' + str(_position) + '\n'
        self.ser.write(command.encode())

    def MoveToAngle(self, _angle):
        command = 'MOVE' + self.axis + 'TOANGLE,' + str(_angle) + '\n'
        self.ser.write(command.encode())

    def MoveAngle(self, _angle):
        command = 'MOVE' + self.axis + 'ANGLE,' + str(_angle) + '\n'
        self.ser.write(command.encode())

    def Move(self, _steps):
        command = 'MOVE' + self.axis + ',' + str(_steps) + '\n'
        self.ser.write(command.encode())

    def SetSpeed(self, _speed):
        command = 'SET' + self.axis + 'SPEED,' + str(_speed) + '\n'
        self.ser.write(command.encode())

    def SetAcceleration(self, _acceleration):
        command = 'SET' + self.axis + 'ACCEL,' + str(_acceleration) + '\n'
        self.ser.write(command.encode())

    def Home(self):
        command = 'HOME' + self.axis + ',\n'
        self.ser.write(command.encode())

    def ReturnToOrigin(self):
        command = 'RETURN' + self.axis + ',\n'
        self.ser.write(command.encode())
    
    def MoveToMax(self):
        command = 'MOVE' + self.axis + 'TOMAX,\n'
        self.ser.write(command.encode())

class StepperControllerInterface:
    ser = None
    def __init__(self, _comPort, _baudRate, _stepperXSpeed, _stepperYSpeed, _stepperXAcceleration, _stepperYAcceleration, _stepperXHomeSpeed, _stepperYHomeSpeed,
                  _stepperXHomeAcceleration, _stepperYHomeAcceleration, _stepperXHomeOffset, _stepperYHomeOffset, _xmax, _ymax):
        try: 
            self.ser = serial.Serial(_comPort, _baudRate, timeout=1)
        except:
            print ("!!! MOTOR CORTEX ERROR: Could not connect to stepper controller")
            return
        
        time.sleep(2)
        self.StepperX = StepperMotor(self.ser, 'X')
        self.StepperY = StepperMotor(self.ser, 'Y')
        command = "INIT," + str(_stepperXSpeed) + "," + str(_stepperYSpeed) + "," + str(_stepperXAcceleration) + "," + str(_stepperYAcceleration) + ","+ str(_stepperXHomeSpeed) + "," + str(_stepperYHomeSpeed) + "," + str(_stepperXHomeAcceleration) + "," + str(_stepperYHomeAcceleration) + "," + str(_stepperXHomeOffset) + "," + str(_stepperYHomeOffset) + "," + str(_xmax) + "," + str(_ymax) + "\n"
        self.ser.write(command.encode())
    def Close(self):
        self.ser.close()
    def ReturnToOrigin(self):
        self.StepperX.ReturnToOrigin()
        self.StepperY.ReturnToOrigin()
    def Home(self):
        self.StepperX.Home()
        self.StepperY.Home()
    def CheckForCommands(self):
        if self.ser.in_waiting > 0:
            self.ser.readline().decode('utf-8').rstrip()
