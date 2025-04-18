import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Simulated small image/mask dataset (e.g., 128x128)
images = np.random.rand(50, 128, 128, 3)
masks = np.random.randint(0, 2, (50, 128, 128, 1))

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D()(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, skip])
    return conv_block(x, filters)

def build_unet():
    inputs = layers.Input((128, 128, 3))
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    bottleneck = conv_block(p2, 256)
    d1 = decoder_block(bottleneck, f2, 128)
    d2 = decoder_block(d1, f1, 64)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d2)
    return models.Model(inputs, outputs)

model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(images, masks, epochs=3)