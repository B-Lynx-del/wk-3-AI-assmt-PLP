"""
BUGGY CODE CHALLENGE - Find and fix the errors!
This TensorFlow code has 5 intentional bugs.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Bug 1: Wrong input shape
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(28,)),  # Should be (784,)
    keras.layers.Dense(10, activation='softmax')
])

# Bug 2: Wrong loss function for multiclass classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Should be 'sparse_categorical_crossentropy'
    metrics=['accuracy']
)

# Bug 3: Dimension mismatch in data
X_train = np.random.rand(100, 784)
y_train = np.random.randint(0, 10, (100, 1))  # Should be (100,) not (100, 1)

# Bug 4: Wrong number of epochs (not technically a bug, but inefficient)
history = model.fit(X_train, y_train, epochs=1, batch_size=32)

# Bug 5: Incorrect prediction shape handling
X_test = np.random.rand(10, 784)
predictions = model.predict(X_test)
predicted_classes = predictions.argmax()  # Should be axis=1

print(f"Predicted classes shape: {predicted_classes.shape}")