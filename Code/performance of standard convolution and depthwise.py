import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train, x_test = x_train / 255.0, x_test / 255.0

# Standard CNN
standard = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(100, activation='softmax')
])

# Depthwise separable CNN
depthwise = models.Sequential([
    layers.SeparableConv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    layers.SeparableConv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(100, activation='softmax')
])

for model in [standard, depthwise]:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training standard CNN")
standard.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

print("Training depthwise CNN")
depthwise.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
