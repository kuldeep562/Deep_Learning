import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf

# Load and prepare Iris dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# From Scratch MLP with 1 hidden layer
np.random.seed(42)
input_size = X.shape[1]
hidden_size = 5
output_size = 3

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(pred, true):
    return -np.mean(np.sum(true * np.log(pred + 1e-9), axis=1))

# Forward pass
Z1 = X_train @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

# Loss
loss = cross_entropy(A2, Y_train)
print(f"Initial loss (from scratch): {loss:.4f}")

# Backpropagation
dZ2 = A2 - Y_train
dW2 = A1.T @ dZ2 / X_train.shape[0]
db2 = np.mean(dZ2, axis=0, keepdims=True)

dA1 = dZ2 @ W2.T
dZ1 = dA1 * relu_deriv(Z1)
dW1 = X_train.T @ dZ1 / X_train.shape[0]
db1 = np.mean(dZ1, axis=0, keepdims=True)

print("Backprop gradients (W1, W2) computed.")

# === Verification using TensorFlow autograd ===

X_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_tf = tf.convert_to_tensor(Y_train, dtype=tf.float32)

class SimpleMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = SimpleMLP()

with tf.GradientTape() as tape:
    preds = model(X_tf)
    loss_tf = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y_tf, preds))

grads = tape.gradient(loss_tf, model.trainable_variables)

print(f"Loss (TF): {loss_tf.numpy():.4f}")
print("Autograd gradients (TF) extracted.")

# Compare shapes of scratch vs TF
print(f"Scratch dW1 shape: {dW1.shape}, TF dW1 shape: {grads[0].numpy().shape}")
print(f"Scratch dW2 shape: {dW2.shape}, TF dW2 shape: {grads[2].numpy().shape}")








