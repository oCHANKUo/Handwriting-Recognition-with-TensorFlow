import tensorflow as tf

model = tf.keras.models.load_model("saved_model_emnist_byclass/best.h5")
model.summary()