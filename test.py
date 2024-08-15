import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Try importing Keras from TensorFlow
from tensorflow import keras
print("Keras version (via TensorFlow):", keras.__version__)

# Build and summarize a simple model to test
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()