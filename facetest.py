import cv2
from ultralytics import YOLO
import lib.CameraInterface as CameraInterface
import time

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model = YOLO("yolov8n.pt")

# Initialize trackers using KCF
trackers = cv2.legacy.MultiTracker_create()

# Start video capture
camera = CameraInterface.CameraInterface(0)

while True:
    # Capture frame-by-frame
    frame = camera.getFrame()
    if frame is None:
        break

    frame = camera.resizeFrame(frame, 640, 480)

    # Update the tracking result
    success, boxes = trackers.update(frame)

    # Draw the tracked objects
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    # Run YOLO detection every N frames or if no trackers
    if len(trackers.getObjects()) == 0:
        detections = model(frame)[0]

        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            # Initialize KCF Tracker and add it to the MultiTracker
            tracker = cv2.legacy.TrackerKCF_create()
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            trackers.add(tracker, frame, bbox)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
camera.release()