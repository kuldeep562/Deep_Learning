import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import cv2

# Load CIFAR-10 and convert to grayscale
(x_train, _), _ = cifar10.load_data()
img = cv2.cvtColor(x_train[0], cv2.COLOR_RGB2GRAY)

# Define kernels
kernels = {
    "Edge Detection": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Box Blur": np.ones((3, 3)) / 9
}

def apply_kernel(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 1 + len(kernels), 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

for i, (name, kernel) in enumerate(kernels.items()):
    result = apply_kernel(img, kernel)
    plt.subplot(1, 1 + len(kernels), i + 2)
    plt.title(name)
    plt.imshow(result, cmap='gray')

plt.tight_layout()
plt.show()
