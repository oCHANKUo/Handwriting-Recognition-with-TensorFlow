import tensorflow as tf
import os

MODEL_DIR = "saved_model_emnist_byclass"

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best.keras'))

model.export(MODEL_DIR)  
print(f"Model saved to {MODEL_DIR}")