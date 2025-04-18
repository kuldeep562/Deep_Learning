import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define model builder for tuning
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=256, step=32), activation='relu'))
    if hp.Boolean('add_layer'):
        model.add(Dense(units=hp.Int('units2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Tune the learning rate for the optimizer
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Set up tuner (Random Search)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='fashion_mnist_tuning'
)

# Run tuning
tuner.search(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)

# Best model
best_model = tuner.get_best_models(1)[0]
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\nBest Model Test Accuracy: {test_acc:.4f}")

# Print best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
print(f"Best Hyperparameters: units1={best_hp.get('units1')}, "
      f"add_layer={best_hp.get('add_layer')}, "
      f"units2={best_hp.get('units2', 'N/A')}, "
      f"learning_rate={best_hp.get('learning_rate')}")
