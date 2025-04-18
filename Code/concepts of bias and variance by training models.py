import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(0)
X = np.sort(np.random.rand(100, 1) * 2 - 1, axis=0)
y = np.sin(np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate models of different polynomial degrees
train_errors = []
val_errors = []
degrees = list(range(1, 15))

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

# Plot bias-variance tradeoff
plt.plot(degrees, train_errors, label='Training Error', marker='o')
plt.plot(degrees, val_errors, label='Validation Error', marker='o')
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.title("Bias vs Variance")
plt.legend()
plt.grid(True)
plt.show()
