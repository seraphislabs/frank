import cv2
from ultralytics import YOLO
import lib.CameraInterface as CameraInterface
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.7
PERSON_CLASS_ID = 0
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model = YOLO("yolov8n.pt")
camera = CameraInterface.CameraInterface(0)

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=10)

while True:
    frame = camera.getFrame()
    if frame is None:
        break

    #frame = camera.resizeFrame(frame, 640, 480)

    # Run YOLO detection
    detections = model(frame)[0]

    # Prepare detections for DeepSORT
    deepsort_detections = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = int(data[5])

        if class_id == PERSON_CLASS_ID and float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            deepsort_detections.append((bbox, confidence, class_id))

    # Update DeepSORT tracks
    tracks = tracker.update_tracks(deepsort_detections, frame=frame)

    # Draw the tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), GREEN, 2)
        cv2.putText(frame, str(track_id), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)

    # Show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()
