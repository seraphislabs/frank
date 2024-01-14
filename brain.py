import lib.globals as brain
import lib.CameraInterface as CameraInterface
from lib.VisualCortex import VisualCortex
from spine import Spine
import tkinter as tk

print("###### Booting Brain ######")
brain.spine = Spine()
print ("###### Brain Booted ######")

print("###### Running Brain ######")

try:
    while brain.spine.shared_flag[0] is True:
        brain.spine.visualCortex.update()
        #brain.spine.motorCortex.update()

except KeyboardInterrupt:
    print("Ctrl+C pressed. Stopping...")

brain.spine.release()
print("Program exited.")