import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Boston Housing Dataset
data = load_boston()
X = data.data
y = data.target.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
X = np.c_[np.ones((X.shape[0], 1)), X]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression from scratch
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1], 1))
        m = X.shape[0]
        for i in range(self.epochs):
            gradients = 2/m * X.T @ (X @ self.theta - y)
            self.theta -= self.lr * gradients

    def predict(self, X):
        return X @ self.theta

# Train model
model = LinearRegressionScratch(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = np.mean((y_pred - y_test)**2)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Plotting first feature against target for visualization
plt.scatter(X_test[:, 1], y_test, color='blue', label='True')
plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Prediction')
plt.show()
