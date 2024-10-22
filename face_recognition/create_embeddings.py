
import sys
import argparse

import numpy as np
import cv2 as cv
import os
import pickle

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

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
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='../face_detection/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.5,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)


# Fungsi untuk mengekstrak embeddings
def extract_embeddings(dataset_path):
    known_face_embeddings = {}
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        embeddings = []
        print(f"Proses embedding untuk: {person_name}")
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img = cv.imread(image_path)
            if img is None:
                print(f"Gagal membaca gambar: {image_path}")
                continue
            h, w, _ = img.shape

            # Inference
            detector.setInputSize([w, h])
            # Deteksi wajah menggunakan SFace
            faces = detector.infer(img)
            
            if faces.shape[0] > 0:  # Pastikan ada wajah terdeteksi
                face = faces[0][:4]  # Ambil bounding box wajah
                embedding = recognizer.infer(img, face).flatten()  # Ekstrak embedding
                embeddings.append(embedding)
                print(f"Embedding berhasil untuk gambar: {image_name}")
            else:
                print(f"Tidak ada wajah terdeteksi dalam gambar: {image_name}")
        
        if embeddings:
            known_face_embeddings[person_name] = embeddings  # Simpan embeddings untuk orang tersebut
            print(f"Proses embedding selesai untuk: {person_name} dengan {len(embeddings)} embedding(s) ditemukan.")
        else:
            print(f"Tidak ada embedding yang berhasil untuk: {person_name}") 
    
    return known_face_embeddings

# Ekstrak embeddings dari dataset
dataset_path = 'ug'  # Ganti dengan path dataset kamu
known_face_embeddings = extract_embeddings(dataset_path)

# Simpan embeddings menggunakan pickle
with open('face_embeddings.pkl', 'wb') as f:
    pickle.dump(known_face_embeddings, f)

print("Embeddings wajah berhasil disimpan.")