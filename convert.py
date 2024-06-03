import tensorflow as tf

# Load the model using keras load_model function
model = tf.keras.models.load_model("64x3-CNN.model")

# Save the model to HDF5 format
model.save("64x3-CNN.h5")
