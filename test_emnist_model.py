import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_DIR = "saved_model_emnist_byclass"
IMAGE_PATH = "test_images/A_real.png"
TARGET_LETTER = 'A'

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best.keras'))

def get_index_from_character(char):
    if char.isdigit():
        return int(char)
    elif char.isupper():
        return ord(char) - ord('A') + 10
    elif char.islower():
        return ord(char) - ord('a') + 36
    return -1

img = Image.open(IMAGE_PATH).convert('L')
img = img.resize((28, 28), Image.LANCZOS)
img_array = np.array(img).astype('float32') / 255.0
img_array = np.transpose(img_array)
img_array = np.fliplr(img_array)
img_array = np.expand_dims(img_array, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array, verbose=0)[0]
target_idx = get_index_from_character(TARGET_LETTER)
accuracy = predictions[target_idx] * 100

print(f"Accuracy: {accuracy:.2f}%")