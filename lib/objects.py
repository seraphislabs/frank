from lib.utils import *

class Person:
    def __init__(self, trackerID, trackerData, img, frame):
        self.firstName = None
        self.lastName = None
        self.trackerID = trackerID
        self.trackerData = trackerData
        self.img = img
        self.faceEmbedding = None
        self.currentFrame = frame


    def GetImageFromFrame(self, frame):
        x, y, w, h = self.trackerData.bbox
        timg = frame[int(y):int(h), int(x):int(w)]

        if timg.shape[0] == 0 or timg.shape[1] == 0:
            self.img = None
            return

        return timg


class TrackerData:
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id

    def toDeepSort(self):
        return (self.bbox, self.confidence, self.class_id)