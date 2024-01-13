from ultralytics import YOLO
import lib.CameraInterface as CameraInterface
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchvision import models, transforms
import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import lib.globals as brain
import datetime
import numpy as np
import time
import uuid

class FaceDetector:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, device='cuda:0')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.preProcess = transforms.Compose([
                        transforms.Resize((160, 160)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
        self.resnet.eval()
        pass

class YoloInterface:
    model = None
    detections = None
    confidence_threshold = None

    def __init__(self, _model="yolov8n.pt", _confidence_threshold=0.6):
        torch.cuda.set_device(0)
        self.confidence_threshold = _confidence_threshold
        self.model = YOLO(_model)
        self.model.fuse()

    def setConfidenceThreshold(self, _confidence_threshold):
        self.confidence_threshold = _confidence_threshold

    def getDetectionsByClass(self, detections, class_id):
        if detections is None:
            return

        returnDetections = []

        for detection in detections.boxes.data.tolist():
            if detection[5] == class_id and detection[4] >= self.confidence_threshold:
                xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                newTrackerObject = TrackerData(bbox, detection[4], detection[5])
                returnDetections.append(newTrackerObject.toDeepSort())

        return returnDetections
    
    def detect(self, frame):
        detections = self.model(frame, augment=True, imgsz=[960, 544], verbose=False, conf=0.25)[0]
        return detections
    
class DeepSortTrackerInterface:
    tracker = None

    def __init__(self, max_age=10):
        #self.tracker = DeepSort(max_age=max_age, nn_budget=None)
        self.tracker = DeepSort(max_age=max_age, n_init=4, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=None, override_track_class=None, 
                                embedder="mobilenet", bgr=True, embedder_gpu=True, embedder_model_name=None, embedder_wts=None, polygon=False, today=None)
        pass

    def syncTracks(self, detections, frame):
        if (detections is None or len(detections) == 0):
            return None
        return self.tracker.update_tracks(detections, frame=frame)

    def drawTracksOnFrame(self, frame, trackedObjects):
        if trackedObjects is None:
            return

        for track in trackedObjects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0,255,0), 2)
            cv2.putText(frame, str(track_id), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

class VisualCortex:
    lastFrame = None
    current_frame = None
    yolo = None
    deepsort = None
    camera = None
    memory = None
    faceDetector = None

    #trackedObjects = []

    trackedPeople = []

    faceEmbeddings = []
    faceEmbeddingsIDs = []

    currentlyTracking = None

    canvas = None

    def __init__(self, memory, max_age=10, camera_id=0):
        self.memory = memory
        model, fps = self.initFromMemory()

        print("** Visual Cortex: Initializing visual cortex")
        print("=> max_age: " + str(max_age) + " model: " + str(model) + " camera_id: " + str(camera_id) + " fps: " + str(fps))
        self.camera = CameraInterface.CameraInterface(camera_id, fps)
        self.yolo = YoloInterface(model)
        self.deepsort = DeepSortTrackerInterface(max_age)
        self.base_path = os.path.dirname(__file__)
        self.faceDetector = FaceDetector()
        self.currentlyTracking = None

        self.getKnownEmbeddings()

        print("=> Visual Cortex Initialized")

        time.sleep(5)
        pass

    def initFromMemory(self):
        sql_query = "SELECT * FROM `visual_cortex` WHERE `id` = %s"
        rows = self.memory.longterm.execute_query(sql_query, ('1',))
        
        return str(rows[0]['model']), int(rows[0]['fps'])

    # Called every Frame (Heartbeat)
    def update(self):
        mode = self.GetMode()
        frame = self.camera.getFrame()

        if frame is None:
                return
        
        self.current_frame = frame
        
        # Fetch the known people's face embeddings from the server
        # Run First pass object detection
        detections = self.yolo.detect(frame)
        peopleDetections = self.yolo.getDetectionsByClass(detections, 0)
        # Update tracking model
        trackedPeople = self.deepsort.syncTracks(peopleDetections, frame)
        # Update self.trackedPeople to be up to date with current trackings and clear old ones
        self.refreshTrackedPeople(trackedPeople)
        # Update people with new images and faces
        self.updatePeople(frame)
        # Add all trackedPeople's embeddings to the mater embedding vs database id list
        self.updateEmbeddings()
        # Compare list of all people to list of known embeddings
        self.updateIdentities()

        self.trackPerson()
        self.pushToUI()

        #self.showEyes()
        self.lastFrame = frame

    def trackPerson(self):
        if self.currentlyTracking is not None:
            print ("Currently tracking name: " + str(self.currentlyTracking.firstName))
            return
        
        potentialTrackees = []
        potentialTrackeesIds = []
        
        for person in self.trackedPeople:
            if person.trackerData is None:
                continue

            if person.face is None:
                continue

            if person.face.identity is None:
                continue

            rectWidth = person.trackerData.bbox[2]
            potentialTrackees.append(rectWidth)
            potentialTrackeesIds.append(self.trackedPeople.index(person))

        if len(potentialTrackees) > 0:
            sorted_pairs = sorted(zip(potentialTrackees, potentialTrackeesIds), reverse=True)
            sorted_potentialTrackees, sorted_potentialTrackeesIds = zip(*sorted_pairs)
            sorted_potentialTrackees = list(sorted_potentialTrackees)
            sorted_potentialTrackeesIds = list(sorted_potentialTrackeesIds)

            self.currentlyTracking = self.trackedPeople[sorted_potentialTrackeesIds[0]]

    def pushToUI(self):
        brain.spine.Queue_TrackedPeople.put(self.trackedPeople)

    def getKnownEmbeddings(self):
        #self.faceEmbeddings, self.faceEmbeddingsIDs = brain.spine.memory.shortterm.get_face_embeddings()
        sql = "SELECT * FROM `people` WHERE `faceEmbedding` IS NOT NULL"
        rows = self.memory.longterm.execute_query(sql)

        for row in rows:
            embedding_np = pickle.loads(row['faceEmbedding'])
            embedding_tensor = torch.from_numpy(embedding_np)
            self.faceEmbeddings.append(embedding_tensor)
            self.faceEmbeddingsIDs.append(row['id'])
        pass

    def updateIdentities(self):
        for person in self.trackedPeople:
            setIdentity = False
            if person.face is None:
                continue

            if person.face.identity is not None:
                continue

            if person.face.img is None:
                continue

            if person.face.face_embedding is None:
                continue

            new_embedding = person.face.face_embedding

            print("Comparing new embedding to " + str(len(self.faceEmbeddings)) + " known embeddings")
            for i, embedding in enumerate(self.faceEmbeddings):
                distance = torch.cosine_similarity(embedding, new_embedding, dim=0)
                #distance = torch.dist(embedding, new_embedding, p=2)
                if distance >= 0.60:
                    person.face.setIdentity(self.faceEmbeddingsIDs[i])
                    person.faceFails = 0
                    setIdentity = True
                    person.getFromDatabase(self.faceEmbeddingsIDs[i])
                    print("Match found: " + str(person.face.identity))
                    break

            print("Checked embeddings")

            if not setIdentity:
                print("No match found, fail attempt " + str(person.faceFails) + " of " + str(person.maxFaceFails))
                person.faceFails += 1
                if person.faceFails < person.maxFaceFails:
                    print("Getting new face")
                    person.face = None
                else:
                    person.faceFails = 0
                    print("Adding new identity")
                    # We've taken 10 pictures and cant find a match. Add new identity to database
                    # TODO: Tie into asking what your name is system
                    #person.addToDatabase()
                    pass

    def updateEmbeddings(self):
        processList = []
        processListObjects = []
        print("Updating embeddings")
        for person in self.trackedPeople:
            if person.face is None:
                continue
        
            if person.face.identity is not None:
                continue

            if person.face.img is None:
                continue

            if person.face.face_embedding is not None:
                continue

            pil_frame = Image.fromarray(cv2.cvtColor(person.face.img, cv2.COLOR_BGR2RGB))
            processList.append(pil_frame)
            processListObjects.append(person.face)
        

        if len(processList) == 0:
            return
        
        faceTensors = [self.faceDetector.preProcess(face) for face in processList]
        faceTensors = torch.stack(faceTensors)

        with torch.no_grad():
            embeddings = self.faceDetector.resnet(faceTensors)

        for i, face in enumerate(embeddings):
            processListObjects[i].face_embedding = face
            tempUUID = str(uuid.uuid1())

    def updatePeople(self, frame):
        for person in self.trackedPeople:
            person.update(frame)

    def refreshTrackedPeople(self, trackedObjects):
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
                    if person.trackingID == track_id:
                        # Update existing person
                        person.trackerData = TrackerData(ltrb, 0, 0)
                        retainedObjects.append(person)
                        found = True
                        break

                if not found:
                    newPerson = Person()
                    newPerson.trackingID = track_id
                    newPerson.trackerData = TrackerData(ltrb, 0, 0)
                    self.trackedPeople.append(newPerson)
                    retainedObjects.append(newPerson)

        # Remove objects not in retainedObjects
        self.trackedPeople = [p for p in self.trackedPeople if p in retainedObjects]
        
        if self.currentlyTracking is not None:
            if self.currentlyTracking not in self.trackedPeople:
                self.currentlyTracking = None
            
    def showEyes(self):
        #cv2.imshow("Frame", self.current_frame)
        for people in self.trackedPeople:
            if people.img is not None:
                cv2.imshow("Tracked Person", people.img)
                if people.face is not None:
                    cv2.putText(self.current_frame, str(people.trackingID) + " Identified: " + str(people.face.identity), (int(people.trackerData.bbox[0]+16), int(people.trackerData.bbox[1]+25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    def release(self):
        self.camera.release()
        pass

    def GetMode(self):
        return self.memory.shortterm.get_value(0, "mode")


class TrackerData:
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id

    def toDeepSort(self):
        return (self.bbox, self.confidence, self.class_id)
        
class Hand:
    def __init__(self, _bbox, _confidence, _class_id):
        self.trackingID = None
        self.trackerData = TrackerData(_bbox, _confidence, _class_id)
        self.handType = None

class Face:
    def __init__(self, img):
        self.img = img
        self.face_embedding = None
        self.identity = None

    def setIdentity(self, identity):
        self.identity = identity

class Person:
    def __init__(self):
        self.trackingID = None
        self.trackerData = None
        self.firstName = None
        self.lastName = None
        self.permissions = None
        self.faceFails = 0
        self.maxFaceFails = 10
        self.xOffset = 0
        self.yOffset = 0

        self.face = None
        self.hands = []
        self.img = None

        self.timeUpdateFrequency = 0.2
        self.lastImgUpdate = datetime.datetime.now()
        self.lastFaceUpdate = datetime.datetime.now()

    def release(self):
        pass

    def updateOffsets(self, frame):
        bbox = self.trackerData.bbox
        if bbox is None:
            return
        
        left, top, right, bottom = bbox
        center_x = left + (right - left) / 2  # Center x-coordinate
        twentyPercentDown = (bottom - top) * 0.2
        adjusted_top_y = top + twentyPercentDown  # 20% down from the top of the bbox

        frame_center_x = frame.shape[1] / 2  # Center x of the frame
        frame_center_y = frame.shape[0] / 2  # Center y of the frame

        self.xOffset = center_x - frame_center_x
        self.yOffset = adjusted_top_y - frame_center_y

    def update(self, frame):
        self.updateOffsets(frame)
        self.checkImage(frame)
        self.checkFace(frame)

    def getFromDatabase(self, databaseIndex):
        sql = "SELECT * FROM `people` WHERE `id` = %s"
        rows = brain.spine.memory.longterm.execute_query(sql, (databaseIndex,))
        self.firstName = rows[0]['firstName']
        self.lastName = rows[0]['lastName']
        embedding_np = pickle.loads(rows[0]['faceEmbedding'])
        embedding_tensor = torch.from_numpy(embedding_np)
        self.face.face_embedding = embedding_tensor
        self.face.identity = databaseIndex

    def addToDatabase(self):
        if self.face is None or self.face.face_embedding is None:
            print ("Error: Cannot add person to database without a face")
            return
        
        sql = "INSERT INTO `people` (`firstName`, `lastName`, `faceEmbedding`) VALUES (%s, %s, %s)"
        print("Adding new person to database")
        embedding_np =  self.face.face_embedding.cpu().detach().numpy()
        serialized = pickle.dumps(embedding_np)

        params = (self.firstName, self.lastName, serialized)
        lastID = brain.spine.memory.longterm.execute_query(sql, params)
        self.face.identity = lastID

        brain.spine.memory.shortterm.db[1].set(serialized, lastID)
        brain.spine.visualCortex.faceEmbeddings.append(self.face.face_embedding)
        brain.spine.visualCortex.faceEmbeddingsIDs.append(lastID)

    def checkImage(self, frame):
        elapsed_time = datetime.datetime.now() - self.lastImgUpdate
        if elapsed_time.total_seconds() > self.timeUpdateFrequency or self.img is None:
            self.lastImgUpdate = datetime.datetime.now()
            self.img = self.getImage(frame)

    def checkFace(self, frame):
        if self.face is not None:
            if self.face.identity is not None:
                return
        
        elapsed_time = datetime.datetime.now() - self.lastFaceUpdate
        if elapsed_time.total_seconds() > self.timeUpdateFrequency or self.face is None:
            self.lastFaceUpdate = datetime.datetime.now()
            gottenFace = self.getFace()
            if gottenFace is not None:
                self.face = gottenFace

    def getImage(self, frame):
        x, y, w, h = self.trackerData.bbox
        timg = frame[int(y):int(y+h), int(x):int(x+w)]

        if timg.shape[0] == 0 or timg.shape[1] == 0:
            self.img = None
            return

        return timg
    
    def getFace(self):
        if self.img is None:
            return None

        pil_frame = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

        # Check if the PIL image is valid
        if pil_frame is None or pil_frame.size[0] == 0 or pil_frame.size[1] == 0:
            return None
        
        # Detect faces
        try:
            boxes, _ = brain.spine.visualCortex.faceDetector.mtcnn.detect(pil_frame)
        except Exception as e:
            print(f"Error: {e}")
            return None

        # Check if any boxes are detected
        if boxes is None or len(boxes) == 0:
            return None

        # Process the first detected face
        left, top, right, bottom = boxes[0]
        face = pil_frame.crop((left, top, right, bottom))
        rgb_face = np.array(face)
        bgr_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)

        # Update or create a new Face object
        if self.face is None:
            self.face = Face(bgr_face)
        else:
            self.face.img = bgr_face

        return self.face