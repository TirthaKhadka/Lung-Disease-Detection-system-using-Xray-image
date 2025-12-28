import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load("/Volumes/External/Pneumonia_Detection_system/backend/BackendForML/saved_model_dir")


# Check the input and output signatures
print("Model input signature:", model.signatures['serving_default'].inputs)
print("Model output signature:", model.signatures['serving_default'].outputs)


