import lib.globals as brain
import lib.appui as ui
import lib.CameraInterface as CameraInterface
from lib.VisualCortex import VisualCortex
from lib.Memory import MemoryInterface
from lib.MotorCortex import MotorCortex
import threading
import tkinter as tk
import queue

class Spine:

    def __init__(self, camera_id=0):
        print ("**** Spine: Initializing spine")
        self.memory = MemoryInterface()

        self.visualCortex = VisualCortex(self.memory, camera_id=camera_id, max_age=12)
        self.motorCortex = MotorCortex("/dev/ttyUSB0")
        self.shared_flag = [True,]

        self.Queue_TrackedPeople = queue.Queue()
        self.appUI = ui.AppUI(self.Queue_TrackedPeople, self.shared_flag)

        self.running = True
        self.setMode("idle")
        print ("=> Spine is initialized")
        pass

    def stop(self):
        self.running = False

    def release(self):
        print("** Spine: Releasing spine")
        self.visualCortex.release()
        self.memory.release()
        print("=> Spine is released")
        pass

    def setMode(self, mode):
        self.memory.shortterm.set_value(0,"mode", mode)
        pass