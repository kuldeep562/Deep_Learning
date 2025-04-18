import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST grayscale images
(x_train, _), _ = mnist.load_data()
image = x_train[0]  # take the first image (28x28)

# Define basic 2D convolution (no padding, stride = 1)
def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output_shape = (
        image.shape[0] - kernel.shape[0] + 1,
        image.shape[1] - kernel.shape[1] + 1,
    )
    output = np.zeros(output_shape)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(region * kernel)

    return output

# Define a sample kernel (e.g., edge detection)
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply convolution
convolved_image = convolve2d(image, kernel)

# Plot original and convolved image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Convolved Image")
plt.imshow(convolved_image, cmap='gray')

plt.tight_layout()
plt.show()
