import lib.globals as brain
from lib.objects import Person, TrackerData
from lib.CameraInterface import CameraInterface
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from lib.utils import *
import torch

class VisualCortex:
    def __init__(self, camera=0, fps=7, max_age=10):
        TextUtils.Debug("Initializing Visual Cortex... ", True)
        self.camera = CameraInterface(camera, fps)

        self.deepsort = DeepSort(max_age=max_age, embedder_gpu=True, n_init=4, max_cosine_distance=0.3, nn_budget=None)
        self.faceDetection = MTCNN(
                        image_size=160,  # size of the cropped face image
                        margin=0,        # margin around the detected face
                        min_face_size=10,  # minimum size of faces to detect
                        thresholds=[0.6, 0.7, 0.7],  # thresholds for P-Net, R-Net, and O-Net
                        factor=0.709,  # scale factor for the image pyramid
                        post_process=True,
                        device='cuda'
                    )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.preProcess = transforms.Compose([
                        transforms.Resize((160, 160)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

        self.trackedPeople = []
        self.currentFrame = None
        self.currentlyTracking = None
        TextUtils.Debug("Done Initializing Visual Cortex\n")

    def draw_bounding_box(self, img, box, color, label):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Internal Methods
    def PushToUI(self):
        brain.spine.Queue_TrackedPeople.put(self.trackedPeople)

    def ScanForFaces(self):
        if self.currentFrame is None or self.currentFrame.shape[0] == 0 or self.currentFrame.shape[1] == 0:
            return [], []
        
        pil_frame = Image.fromarray(cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2RGB))

        # Check if the PIL image is valid
        if pil_frame is None:
            return [], []

        try:
            boxes, probs, landmarks = self.faceDetection.detect(pil_frame, landmarks=True)
        except Exception as e:
            print(f"Error: {e}")
            return [], []
        
        if boxes is None:
            return [], []
        
        faces = []
        newBoxes = []
        for box, landmark in zip(boxes, landmarks):
            #aligned_face = self.align_face(pil_frame, landmark)
            faces.append(pil_frame)

            # Convert the box to ltrb format
            box = box.tolist()
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]

            newBoxes.append((box, 0, 0))
            #print("Aligned face shape:", aligned_face.size, "NewBBox:", box)

        faceTensors = [self.preProcess(face) for face in faces]
        faceTensors = torch.stack(faceTensors)

        with torch.no_grad():
            faceEmbeddings = self.resnet(faceTensors)

        return newBoxes, faceEmbeddings
        
    def TrackFaces(self, boxes, faceEmbeddings, frame):
        if (len(boxes) != len(faceEmbeddings)):
            return []
        
        tracks = self.deepsort.update_tracks(boxes, embeds=faceEmbeddings, frame=frame)
        #tracks = self.deepsort.update_tracks(boxes, frame=frame)

        results = []
        debug_frame = frame.copy()  # Copy the frame for debugging

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # Top left and bottom right coordinates
            results.append(track)

        return results
    
    def align_face(self, img, landmarks):
        """
        Align the face using the landmarks
        """
        # Define the desired left eye position and the desired face width and height
        desired_left_eye = (0.35, 0.35)
        desired_face_width = 160
        desired_face_height = 160

        # Compute the angle between the eyes
        dY = landmarks[1,1] - landmarks[0,1]
        dX = landmarks[1,0] - landmarks[0,0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        # Determine the scale of the new resulting image by taking the ratio of the distance between the eyes
        # in the current image to the ratio of distance between the eyes in the desired image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0])
        desired_dist *= desired_face_width
        scale = desired_dist / dist

        # Compute the center (x, y)-coordinates between the two eyes in the input image
        eyes_center = ((landmarks[0,0] + landmarks[1,0]) // 2, (landmarks[0,1] + landmarks[1,1]) // 2)

        # Grab the rotation matrix for rotating and scaling the face, then apply the translation
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix to shift the eyes to the center of the image
        tX = desired_face_width * 0.5
        tY = desired_face_height * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        #Apply the affine transformation
        img_array = np.array(img)
        (w, h) = (desired_face_width, desired_face_height)
        output = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC)

        #convert to pil
        return Image.fromarray(output)
    
    def SyncTracksToObjects(self, trackedObjects):
        retainedObjects = []

        if trackedObjects is not None:
            for track in trackedObjects:

                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()

                # Check if the object is already being tracked
                found = False
                for i, person in enumerate(self.trackedPeople):
                    if person.trackerID == track_id:
                        # Update existing person
                        person.trackerData.bbox = ltrb
                        retainedObjects.append(person)
                        found = True
                        break

                if not found:
                    trackData = TrackerData(ltrb, 1, 0)
                    newPerson = Person(track_id, trackData, None, self.currentFrame)
                    self.trackedPeople.append(newPerson)
                    retainedObjects.append(newPerson)

        # Remove objects not in retainedObjects
        self.trackedPeople = [p for p in self.trackedPeople if p in retainedObjects]

    def UpdatePeople(self):
        for person in self.trackedPeople:  
            person.img = person.GetImageFromFrame(self.currentFrame)
            if person.img is None:
                continue

    def update(self):
        frame = self.camera.getFrame()
        self.currentFrame = frame

        if frame is None:
            return
        
        if frame.mean() < 50:
            print("Frame is too dark")
            return
        
        boxes, faceEmbeddings = self.ScanForFaces()
        tracks = self.TrackFaces(boxes, faceEmbeddings, frame)
        self.SyncTracksToObjects(tracks)
        self.UpdatePeople()
    
        self.PushToUI()