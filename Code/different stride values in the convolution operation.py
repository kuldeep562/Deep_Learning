import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import svhn
import cv2

# Load SVHN-style image manually from CIFAR or any sample RGB image
from tensorflow.keras.datasets import cifar10
(x_train, _), _ = cifar10.load_data()
img = cv2.cvtColor(x_train[0], cv2.COLOR_RGB2GRAY)

def convolve_stride(image, kernel, stride=1):
    k = kernel.shape[0]
    out_dim = (image.shape[0] - k) // stride + 1
    result = np.zeros((out_dim, out_dim))
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            region = image[i*stride:i*stride+k, j*stride:j*stride+k]
            result[i, j] = np.sum(region * kernel)
    return result

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

stride_1 = convolve_stride(img, kernel, stride=1)
stride_2 = convolve_stride(img, kernel, stride=2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Stride = 1")
plt.imshow(stride_1, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Stride = 2")
plt.imshow(stride_2, cmap='gray')
plt.tight_layout()
plt.show()
