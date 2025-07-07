
import sys
import argparse

import numpy as np
import cv2 as cv
import os
import pickle
import albumentations as A
from albumentations.core.composition import OneOf

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


# Augmentasi menggunakan Albumentations
augmentation_pipeline = A.Compose([
    OneOf([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ], p=1.0),
    A.Resize(224, 224)  # Opsional, untuk resize
])

# Fungsi untuk menyimpan gambar augmentasi
def save_augmented_images(image, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)  # Pastikan folder output ada
    for i in range(3):  # Simpan 3 gambar augmentasi
        augmented = augmentation_pipeline(image={"image": image})
        augmented_image = augmented["image"]
        output_file = os.path.join(output_dir, f"{base_filename}_aug_{i+1}.jpg")
        cv.imwrite(output_file, cv.cvtColor(augmented_image, cv.COLOR_RGB2BGR))
        print(f"Saved: {output_file}")

# Fungsi untuk mengekstrak embeddings
def run_augment(dataset_path):
    output_path = 'augmen_face'
    i = 0
    for person_name in [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]:
        person_path = os.path.join(dataset_path, person_name)
        print(f"Proses augment untuk: {person_name}")
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img = cv.imread(image_path)
            if img is None:
                print(f"Gagal membaca gambar: {image_path}")
                continue

            h, w, _ = img.shape
            detector.setInputSize([w, h])
            faces = detector.infer(img)

            base_filename = os.path.splitext(person_name)[0].replace(' ', '_').lower()
            if faces.shape[0] > 0:  # Pastikan ada wajah terdeteksi
                for face in faces:  # Untuk setiap wajah
                    x1, y1, x2, y2 = map(int, face[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    cropped_face = img[y1:y2, x1:x2]  # Crop wajah
                    save_augmented_images(cropped_face, output_path, f"{base_filename}_face_{i+1}")
                    i += 1
            else:
                print(f"Tidak ada wajah terdeteksi dalam gambar: {image_name}")

    

# Ekstrak embeddings dari dataset
dataset_path = 'ug'  # Ganti dengan path dataset kamu
run_augment(dataset_path)
