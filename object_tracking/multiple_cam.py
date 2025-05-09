# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys
import pickle
# from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
# from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
# from deep_sort.application_util.visualization import cv2
from deep_sort_pytorch.deep_sort import DeepSort

from os.path import dirname, join


sys.path.append(os.path.abspath('../face_detection'))
from yunet import YuNet
sys.path.append(os.path.abspath('../face_recognition'))
from sface import SFace


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
ap.add_argument("-m", "--model", required=False,default='MobileNetSSD_deploy.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument('--backend_target', '-bt', type=int, default=1,
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
PERSON_CLASS_ID = CLASSES.index("person")  # 15
backend_id = backend_target_pairs[args["backend_target"]][0]
target_id = backend_target_pairs[args["backend_target"]][1]

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

    if known_face_embeddings is not None:
    # Bandingkan dengan embedding wajah yang sudah dikenal
        for name, embeddings in known_face_embeddings.items():
            for known_embedding in embeddings:
                # similarity = cosine_similarity(face_embedding, known_embedding)
                similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))            
                if similarity > max_similarity:
                    max_similarity = similarity
                    recognized_label = name

    return recognized_label, max_similarity if max_similarity > SIMILARITY_THRESHOLD else 0



# Instantiate SFace for face recognition
recognizer = SFace(modelPath='../face_recognition/face_recognition_sface_2021dec.onnx',
                    disType=0,
                    backendId=backend_id,
                    targetId = target_id)
# Instantiate YuNet for face detection
detector = YuNet(modelPath='../face_detection/face_detection_yunet_2023mar.onnx',
                    inputSize=[320, 320],
                    confThreshold=0.9,
                    nmsThreshold=0.3,
                    topK=5000)


# Load known face embeddings
with open('../face_recognition/face_embeddings_new.pkl', 'rb') as f:
    known_face_embeddings = pickle.load(f)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# Inisialisasi DeepSort
# config = DeepSORTConfig()
tracker = DeepSort(
    model_path="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",  # Path ke model DeepSORT (sesuaikan dengan path model Anda)
    max_dist=0.1,
    min_confidence=0.2,
    nms_max_overlap=0.9,
    max_iou_distance=0.6,
    max_age=2,
    n_init=2,
    nn_budget=90
)
# Inisialisasi video stream 
print("[INFO] starting video stream...")
cap1 = cv2.VideoCapture(4) #kamera person
cap2 = cv2.VideoCapture(0) #kamera face
time.sleep(2.0)
fps = FPS().start()

# Mulai menghitung waktu secara manual
start_time = time.time()
id_face_recognition = {}
id_face = 0
id_to_label = {}
enter_time = {}  # Untuk mencatat waktu masuk
last_seen_time = {}  # Untuk mencatat waktu terakhir terlihat
# Ambang batas untuk menganggap seseorang keluar (dalam detik)
EXIT_THRESHOLD = 5  # 
# Loop melalui frame1 video
while True:
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret or not ret2:
        break
  
    frame1 = imutils.resize(frame1, width=920)
    (h, w) = frame1.shape[:2]

    # Membuat blob dari frame1
    blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    
    detection_boxes = []
    detection_scores = []
    
    # Loop melalui deteksi
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:  # Sesuaikan threshold confidence sesuai kebutuhan Anda
            idx = int(detections[0, 0, i, 1])

            if idx == PERSON_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # print(f"Detected Box: {startX, startY, endX, endY}")
                detection_boxes.append([startX, startY, endX , endY ])  
                detection_scores.append(confidence)

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame1, (startX, startY), (endX, endY), (0, 0, 255), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame1, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                
    # Konversi deteksi ke array NumPy
    detection_boxes =  np.array([[(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)] for (x1, y1, x2, y2) in detection_boxes])
    detection_scores = np.array(detection_scores)
    oids = None
    if not oids:
            oids = [-1] * len(detection_scores)
            
    # Tracking menggunakan DeepSORT
    if len(detection_boxes) > 0 and len(detection_scores) > 0:
        trackers = tracker.update(detection_boxes, detection_scores, oids, frame1)   
        
       
    # Deteksi wajah menggunakan YuNet
    detector.setInputSize((frame2.shape[1], frame2.shape[0]))
    faces = detector.infer(frame2)
    for face in faces:
        x, y, w, h, conf = face[:5].astype(int)
        
            # Pastikan bounding box valid sebelum melakukan face recognition
        if x < 0 or y < 0 or x + w > frame2.shape[1] or y + h > frame2.shape[0]:
            continue
        
        face_roi = frame2[y:y+h, x:x+w]
        if face_roi.size == 0 or w < 10 or h < 10:
            continue
        
        face_embedding = recognizer.infer(frame2, face).flatten()
        if face_embedding.shape != (128,):  # Ganti dengan dimensi embedding yang sesuai
            continue

        # Lakukan pengenalan wajah
        recognized_label, similarity = recognize_face(face_embedding, known_face_embeddings)
        
        if similarity > SIMILARITY_THRESHOLD:
            for track in trackers:
                track_id = int(track[4])
                tx1, ty1, tw, th = map(int, track[:4])
                # if tx1 <= x <= tx1 + tw and ty1 <= y <= ty1 + th:
                if track_id not in id_to_label:
                    id_to_label[track_id] = recognized_label
                    enter_time[track_id] = time.time()
                last_seen_time[track_id] = time.time()
                
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame2, f"{recognized_label} {similarity:.02f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame1, (tx1, ty1), (tw, th  ), (0, 255, 0), 2)
                cv2.putText(frame1, f"ID: {track_id}", (tx1, ty1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
                
                

    #Cek apakah ada orang yang sudah keluar berdasarkan waktu terakhir terlihat
    current_time = time.time()
    for track_id in list(last_seen_time.keys()):
        if current_time - last_seen_time[track_id] > EXIT_THRESHOLD :
            # Ambil waktu masuk atau setel default ke 0 jika tidak ada
            enter_time_value = enter_time.get(track_id, None)
            if enter_time_value is not None:
                # Anggap orang sudah keluar
                print(f"Person ID {track_id} with label {id_to_label.get(track_id, 'Unknown')} has left the room. total time: {current_time - enter_time_value:.2f} seconds")
            
            # Hapus ID dari dictionary yang terkait
            last_seen_time.pop(track_id, None)
            enter_time.pop(track_id, None)
            id_to_label.pop(track_id, None)
            
    # Update FPS counter
    fps.update()

    # Hitung waktu berlalu
    elapsed_time = time.time() - start_time
    fps_estimate = fps._numFrames / elapsed_time

    # Tampilkan FPS yang dihitung manual
    fps_text = "FPS: {:.2f}".format(fps_estimate)
    cv2.putText(frame1, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Person Detection and Tracking (cap1)", frame1)
    cv2.imshow("Face Recognition (cap2)", frame2)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    
    
    

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
cap1.release()  # Gunakan release() untuk VideoCapture
