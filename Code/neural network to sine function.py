import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Generate sine wave data
np.random.seed(0)
X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different architectures
architectures = [
    [64, 64],
    [128, 64, 32],
    [32],
]

for i, arch in enumerate(architectures, 1):
    print(f"\nTraining Model {i} with architecture {arch}...")

    model = Sequential()
    model.add(Dense(arch[0], activation='relu', input_shape=(1,)))
    for units in arch[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    y_pred = model.predict(X_test)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model {i} Test Loss: {test_loss:.5f}")

    # Plot
    X_plot = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    y_true = np.sin(X_plot)
    y_approx = model.predict(X_plot)

    plt.figure()
    plt.plot(X_plot, y_true, label='True Sine', color='blue')
    plt.plot(X_plot, y_approx, label='Predicted', color='red')
    plt.title(f'Sine Approximation - Model {i}')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
