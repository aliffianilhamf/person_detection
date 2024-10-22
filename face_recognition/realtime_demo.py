# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import numpy as np
import cv2 as cv
import os
import pickle
import time
# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

sys.path.append(os.path.abspath('../person_detection'))
from demo_person_detection import visualize
from mp_persondet import MPPersonDet
from sface import SFace

# sys.path.append('../face_detection')
sys.path.append(os.path.abspath('../face_detection'))
from yunet import YuNet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--target', '-t', type=str,
                    help='Usage: Set path to the input image 1 (target face).')
parser.add_argument('--query', '-q', type=str,
                    help='Usage: Set path to the input image 2 (query).')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    
    # Fungsi untuk menghitung Intersection over Union (IoU)
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='../face_detection/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)
  
    person_model = MPPersonDet(modelPath='../person_detection/person_detection_mediapipe_2023mar.onnx',
                           nmsThreshold=0.5,
                           scoreThreshold=0.3,
                           topK=5000,
                           backendId=backend_id,
                           targetId=target_id)

    # Person Tracking data
    person_dict = {}  # To store recognized persons, their entry times, and last seen time
    trackers = [] 
    timeout_duration = 5 * 60  # 5 minutes in seconds
    
    # Load known face embeddings
    with open('face_embeddings.pkl', 'rb') as f:
        known_face_embeddings = pickle.load(f)

   # Pencocokan wajah real-time menggunakan kamera
    cap = cv.VideoCapture(0)

    while True:
        # Baca frame dari kamera
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        faces = detector.infer(frame)
        
        recognized_persons = [] 

        if faces is not None:
            for face in faces:
                x, y, w, h, conf = face[:5].astype(int)

                # Cek jika bounding box valid
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue

                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0 or w < 10 or h < 10:
                    continue

                face_embedding = recognizer.infer(frame, face).flatten()
                if face_embedding.shape != (128,):  # Ganti dengan dimensi embedding yang sesuai
                    continue

                label = "Unknown"
                max_similarity = 0

                # Jika ada embeddings yang dikenal, lakukan pencocokan
                if known_face_embeddings is not None:
                    for known_label, embeddings in known_face_embeddings.items():
                        for known_embedding in embeddings:
                            similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
                            if similarity > max_similarity:
                                max_similarity = similarity
                                label = f"{known_label} ({similarity:.2f})"
                    
                # Gambarkan bounding box wajah dan label
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text_position = (x, y - 10 if y - 10 > 10 else y + 10)
                cv.putText(frame, label, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    
        
            
        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

   
