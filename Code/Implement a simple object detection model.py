import tensorflow as tf
import numpy as np

# Dummy dataset (image + bounding box: [x, y, w, h])
images = np.random.rand(100, 64, 64, 3)
boxes = np.random.rand(100, 4)  # normalize between 0 and 1

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')  # output: bounding box
])

model.compile(optimizer='adam', loss='mse')
model.fit(images, boxes, epochs=5)
