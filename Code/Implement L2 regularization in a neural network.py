import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredError

# Load dataset
boston = load_boston()
X = boston.data
y = boston.target

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model without regularization
model_no_reg = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_no_reg.compile(optimizer='adam', loss='mse')
model_no_reg.fit(X_train, y_train, epochs=100, verbose=0)
mse_no_reg = model_no_reg.evaluate(X_test, y_test, verbose=0)

# Model with L2 regularization
model_l2 = Sequential([
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1)
])
model_l2.compile(optimizer='adam', loss='mse')
model_l2.fit(X_train, y_train, epochs=100, verbose=0)
mse_l2 = model_l2.evaluate(X_test, y_test, verbose=0)

# Results
print(f"Test MSE without L2 Regularization: {mse_no_reg:.4f}")
print(f"Test MSE with L2 Regularization   : {mse_l2:.4f}")
