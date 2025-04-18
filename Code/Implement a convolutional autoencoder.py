import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10  # Placeholder for STL-10

(x_train, _), _ = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0

# Simple encoder-decoder
input_img = tf.keras.Input(shape=(32, 32, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, validation_split=0.1)

# Save encoded features
encoder = models.Model(input_img, encoded)
features = encoder.predict(x_train)
print("Feature shape:", features.shape)
