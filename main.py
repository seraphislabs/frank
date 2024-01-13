import time
import lib.MotorCortex as MotorCortex

#StepperControllerInterface - COM port, baud rate, xSpeed, ySpeed, xAcceleration, yAcceleration, xHomeSpeed, yHomeSpeed, xHomeAcceleration, yHomeAcceleration, xHomeOffset, yHomeOffset, xmax, ymax
sc = MotorCortex.StepperControllerInterface('/dev/ttyUSB0', 250000, 4000, 8000, 2000, 6000, 200, 200, 20000, 20000, -700, -350, 1400, 1800)

time.sleep(3)
sc.StepperY.Home()
sc.StepperX.Home()

sc.Close()