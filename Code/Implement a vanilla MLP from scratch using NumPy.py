import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten input images for MLP
x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

### Part 1: MLP using TensorFlow
print("Training TensorFlow MLP...")

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"TensorFlow MLP Test Accuracy: {test_acc:.4f}")

### Part 2: Simple MLP using NumPy
# For demonstration, we'll use a small sample due to slowness of NumPy-only models
print("Training NumPy MLP...")

sample_size = 1000
X = x_train_flat[:sample_size]
Y = to_categorical(y_train[:sample_size].flatten(), 10)

# Initialize weights
input_dim = X.shape[1]
hidden_dim = 64
output_dim = 10

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy(pred, true):
    return -np.sum(true * np.log(pred + 1e-9)) / pred.shape[0]

# Training loop
lr = 0.1
epochs = 100

for epoch in range(epochs):
    # Forward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss
    loss = cross_entropy(A2, Y)

    # Backward
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1 / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation on test sample
Z1 = x_test_flat[:500] @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

y_pred = np.argmax(A2, axis=1)
y_true = y_test[:500].flatten()
accuracy = np.mean(y_pred == y_true)
print(f"NumPy MLP Test Accuracy (on sample): {accuracy:.4f}")
