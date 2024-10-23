# import necessary packages
from imutils.video import VideoStream, FPS
import numpy as np
import imutils
import time
import cv2
from collections import OrderedDict
from scipy.spatial import distance as dist
import argparse
import os
import sys
from collections import defaultdict
import pickle

sys.path.append(os.path.abspath('../face_detection'))
from yunet import YuNet
sys.path.append(os.path.abspath('../face_recognition'))
from sface import SFace

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        # Initialize next object ID, objects dict, and disappeared counters
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # Register a new object with a unique ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove the object by ID
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # If no bounding boxes, increment disappearance counter for all objects
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # Initialize an array of input centroids
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# Threshold similarity untuk mengenali wajah
SIMILARITY_THRESHOLD = 0.6
# Fungsi untuk menghitung cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def recognize_face(face_embedding,known_face_embeddings):
    recognized_label = "Unknown"
    max_similarity = 0

    # Bandingkan dengan embedding wajah yang sudah dikenal
    for name, embeddings in known_face_embeddings.items():
        for known_embedding in embeddings:
            # similarity = cosine_similarity(face_embedding, known_embedding)
            similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))            
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_label = name

    # Jika similarity melebihi threshold, anggap wajah dikenali
    if max_similarity >= SIMILARITY_THRESHOLD:
        return recognized_label, max_similarity
    else:
        return "Unknown", max_similarity
# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# Argumen parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default='MobileNetSSD.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='MobileNetSSD_deploy.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
args = vars(ap.parse_args())
# Daftar kelas yang benar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Tentukan indeks kelas "person"
PERSON_CLASS_ID = CLASSES.index("person")
print(args)
backend_id = backend_target_pairs[args["backend_target"]][0]
target_id = backend_target_pairs[args["backend_target"]][1]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



# Instantiate SFace for face recognition
recognizer = SFace(modelPath='../face_recognition/face_recognition_sface_2021dec.onnx',
                    disType=0,
                    backendId=backend_id,)
# Instantiate YuNet for face detection
detector = YuNet(modelPath='../face_detection/face_detection_yunet_2023mar.onnx',
                    inputSize=[320, 320],
                    confThreshold=0.9,
                    nmsThreshold=0.3,
                    topK=5000)

# Load known face embeddings
with open('../face_recognition/face_embeddings.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

# Inisialisasi video stream
print("[INFO] starting video stream...")
# Inisialisasi video stream
vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
TIMEOUT_DURATION = 5 * 60  # 5 menit dalam detik

time.sleep(2.0)
fps = FPS().start()

# Inisialisasi centroid tracker
ct = CentroidTracker()
# Centroid tracker untuk melacak object person
person_track_times = defaultdict(lambda: {'entry_time': None, 'last_seen': None, 'recognized_face': None})

# Loop melalui frame video
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    (h, w) = frame.shape[:2]

    # Deteksi wajah menggunakan YuNet
    faces = detector.infer(cv2.resize(frame, (320, 320)))
    
    # Deteksi person menggunakan SSDMobileNet
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    rects = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            if idx == PERSON_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                rects.append((startX, startY, endX, endY))
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    objects = ct.update(rects)

    for face in faces:
        x, y, w, h, conf = face[:5].astype(int)
        # Pastikan bounding box valid sebelum melakukan face recognition
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue
        
        face_roi = frame[y:y+h, x:x+w]
        face_embedding = recognizer.infer(face_roi).flatten()

        # Lakukan pengenalan wajah
        recognized_label, similarity = recognize_face(face_embedding, known_face_embeddings)

        if recognized_label != "Unknown":
            for objectID, centroid in objects.items():
                person_track_times[objectID]['recognized_face'] = recognized_label
                person_track_times[objectID]['last_seen'] = time.time()

                if person_track_times[objectID]['entry_time'] is None:
                    person_track_times[objectID]['entry_time'] = time.time()

        # Tampilkan label pengenalan wajah pada frame
        cv2.putText(frame, f"{recognized_label} ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    current_time = time.time()
    for objectID in list(person_track_times.keys()):
        last_seen = person_track_times[objectID]['last_seen']

        # Jika person sudah tidak terdeteksi dalam waktu yang ditentukan (5 menit)
        if current_time - last_seen > TIMEOUT_DURATION:
            print(f"Person {objectID} keluar ruangan, total waktu: {current_time - person_track_times[objectID]['entry_time']} detik.")
            del person_track_times[objectID]
        else:
            print(f"Person {objectID} masih di dalam ruangan, terakhir terdeteksi {current_time - last_seen} detik yang lalu.")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()

# Cleanup
cv2.destroyAllWindows()
vs.release()