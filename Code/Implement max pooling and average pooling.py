import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, _), _ = fashion_mnist.load_data()
img = x_train[0]

def pool2d(img, size=2, stride=2, mode='max'):
    h, w = img.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    pooled = np.zeros((out_h, out_w))

    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            region = img[i:i+size, j:j+size]
            if mode == 'max':
                pooled[i//stride, j//stride] = np.max(region)
            else:
                pooled[i//stride, j//stride] = np.mean(region)

    return pooled

max_pooled = pool2d(img, mode='max')
avg_pooled = pool2d(img, mode='avg')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Max Pooling")
plt.imshow(max_pooled, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Avg Pooling")
plt.imshow(avg_pooled, cmap='gray')
plt.tight_layout()
plt.show()
