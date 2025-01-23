# import the necessary packages
from typing import Optional

import pytz

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
from datetime import datetime, timezone
from ConsumeApi import PostPersonDuration, PostDetailPersonDuration, UpdateEndTimePersonDuration

from os.path import dirname, join


sys.path.append(os.path.abspath('../face_detection'))
from yunet import YuNet
sys.path.append(os.path.abspath('../face_recognition'))
from sface import SFace


# Valid combinations of backends and targets
backend_target_pairs = [
    #[cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    #[cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    #[cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
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
                        #: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        #: TIM-VX + NPU,
                        #: CANN + NPU
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
SIMILARITY_THRESHOLD = 0.5
# Fungsi untuk menghitung cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def recognize_face(face_embedding,known_face_embeddings):
    global person_name, person_nim
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
                    person_name = recognized_label.replace("_", " ")[0:-16]
                    person_nim = recognized_label[-14:]

    return person_name, person_nim, max_similarity if max_similarity > SIMILARITY_THRESHOLD else 0



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
with open('../face_recognition/face_embeddings.pkl', 'rb') as f:
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
video_path = 'LeftCam.mp4'
vs = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

# Mulai menghitung waktu secara manual
start_time = time.time()
id_to_label = {}
enter_time = {}  # Untuk mencatat waktu masuk
last_seen_time = {}  # Untuk mencatat waktu terakhir terlihat
# Ambang batas untuk menganggap seseorang keluar (dalam detik)
EXIT_THRESHOLD = 60  # 
saved_images = []
# Loop melalui frame video
while True:
    recognized_dir: Optional[str] = None
    person_name: Optional[str] = None
    name_file: Optional[str] = None
    ret, frame = vs.read()
    if not ret:
        break
  
    frame = imutils.resize(frame, width=720)
    (h, w) = frame.shape[:2]

    # Membuat blob dari frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    
    detection_boxes = []
    detection_scores = []
    
    # Loop melalui deteksi
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:  
            idx = int(detections[0, 0, i, 1])

            if idx == PERSON_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # print(f"Detected Box: {startX, startY, endX, endY}")
                detection_boxes.append([startX, startY, endX , endY ])  
                detection_scores.append(confidence)

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                
    # Konversi deteksi ke array NumPy
    detection_boxes =  np.array([
    [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)] for (x1, y1, x2, y2) in detection_boxes
])
    detection_scores = np.array(detection_scores)
    oids = None
    if not oids:
            oids = [-1] * len(detection_scores)
            
    # Tracking menggunakan DeepSORT
    if len(detection_boxes) > 0 and len(detection_scores) > 0:
        trackers = tracker.update(detection_boxes, detection_scores, oids, frame)
    # print(f"Trackers: {trackers}")
        for track in trackers:
            track_id = int(track[4])  # Dapatkan ID dari DeepSORT
            startX, startY, width, height = track[:4]
            # print(f"DeepSORT Detected Box: {startX, startY, width, height}")
            endX, endY = startX + width, startY + height
            
            # Deteksi wajah menggunakan YuNet
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            faces = detector.infer(frame)
            for face in faces:
                x, y, w, h, conf = face[:5].astype(int)
                if startX <= x <= endX and startY <= y <= endY:
                    # Pastikan bounding box valid sebelum melakukan face recognition
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0 or w < 10 or h < 10:
                        continue
                    
                    face_embedding = recognizer.infer(frame, face).flatten()
                    if face_embedding.shape != (128,):  # Ganti dengan dimensi embedding yang sesuai
                        continue

                    # Lakukan pengenalan wajah
                    person_name, person_nim, similarity = recognize_face(face_embedding, known_face_embeddings)
                    # Define the directory where recognized images will be saved
                    recognized_dir = "recognized_faces"

                    if not os.path.exists(recognized_dir):
                        os.makedirs(recognized_dir)
                    person_crop = frame[startY:startY+height, startX:startX+width]
                    # Jika wajah dikenali dan ID belum memiliki label, tambahkan ke dictionary
                    if similarity > SIMILARITY_THRESHOLD:
                        # Cek apakah ID belum memiliki label, tambahkan ke dictionary
                        if track_id not in id_to_label:
                            id_to_label[track_id] = person_name
                            enter_time[track_id] = time.time()  # Catat waktu masuk

                            # Check if in recognized_faces has recognized_label
                            if person_name in saved_images:
                                continue
                            else:
                                saved_images.append(person_name)
                                image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_RGB2BGR)
                                name_file = f"{recognized_dir}/{person_name}.jpg"
                                if name_file in os.listdir(recognized_dir):
                                    continue
                                PostPersonDuration(person_name, track_id)
                                PostDetailPersonDuration(image_rgb, person_nim, person_name, track_id)
                                cv2.imwrite(name_file, person_crop)
                                print(f"Person ID {track_id} with label {person_name} has entered the room.")

                        label = id_to_label.get(track_id, "Unknown")
                        cv2.putText(frame, f"{label} ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Perbarui waktu terakhir terlihat
            last_seen_time[track_id] = time.time()

            # Tampilkan ID pada frame
           
            cv2.rectangle(frame, (startX, startY), (width, height  ), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

    # Cek apakah ada orang yang sudah keluar berdasarkan waktu terakhir terlihat
    current_time = time.time()
    for track_id in list(last_seen_time.keys()):
        if current_time - last_seen_time[track_id] > EXIT_THRESHOLD:
            
            enter_time_value = enter_time.get(track_id, None)
            if enter_time_value is not None:
                # Anggap orang sudah keluar
                # Safely remove an element from the list
                label = id_to_label.get(track_id)

                if label:
                    if label in saved_images:
                        saved_images.remove(label)
                    else:
                        print(f"Label '{label}' not found in saved_images")
                else:
                    print(f"Track ID {track_id} not found in id_to_label")


                indo_time= datetime.now(pytz.timezone('Asia/Jakarta'))
                formatted_time = indo_time.strftime('%Y-%m-%dT%H:%M:%S')
                
                print(f"Person ID {track_id} with label {id_to_label.get(track_id, 'Unknown')} has left the room. total time: {current_time} seconds and trackid : {id_to_label[track_id]}{track_id}")
                UpdateEndTimePersonDuration(f"{id_to_label[track_id]}{track_id}", formatted_time)
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
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Result of Person Detection and Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    
    
    

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.release()  # Gunakan release() untuk VideoCapture
