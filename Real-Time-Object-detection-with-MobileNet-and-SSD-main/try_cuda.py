import cv2
import numpy as np

# Buat gambar dummy
image = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))  # Contoh: ubah ukuran ke 512x512


# Upload ke GPU
gpu_image = cv2.cuda_GpuMat()
try:
    gpu_image.upload(image)
except cv2.error as e:
    print(f"OpenCV CUDA Error: {e}")


# Uji operasi sederhana (invert warna)
gpu_inverted = cv2.cuda.bitwise_not(gpu_image)
result = gpu_inverted.download()

# Tampilkan hasil
cv2.imshow('Inverted Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
