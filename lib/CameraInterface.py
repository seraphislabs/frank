import cv2
import os
import time
import uuid

class CameraInterface:
    def __init__(self, video_source=0, fps=10):
        print("** CameraInterface: Initializing camera interface")
        print("=> video_source: " + str(video_source) + " fps: " + str(fps))
        # Initialize camera interface

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Camera could not be opened")
            return
        
        assert self.cap.isOpened()
        
        time.sleep(9)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        newFPS = fps
        self.cap.set(cv2.CAP_PROP_FPS, newFPS)

        print("=> Camera Initialized")

    def resizeFrame(self, frame, width, height):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def getFrame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if (ret == False):
            return None 
        
        leftFrame = frame[:, :frame.shape[1] // 2]
        return leftFrame

    def getKeypress(self):
        # Check if 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 'q'
        return None
    
    def release(self):
        # Release the capture
        self.cap.release()
        cv2.destroyAllWindows()
        pass

    def runTrainingCapture(self, iterations, imgpath):
        for imgnum in range(iterations):
            print("Collecting image {}".format(imgnum))
            frame = self.getFrame()
            if frame is None:
                break
            imgname = os.path.join(imgpath, str(uuid.uuid1()) + '.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('Frame', frame)
            time.sleep(1)

            if self.camera.getKeypress() == 'q':
                break