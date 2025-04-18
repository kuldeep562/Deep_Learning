import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = tf.image.resize(x_train, (224, 224)) / 255.0, tf.image.resize(x_test, (224, 224)) / 255.0

# Load base VGG16
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze

# Add classifier
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
