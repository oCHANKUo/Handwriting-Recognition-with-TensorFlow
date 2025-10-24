import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

MODEL_DIR = "saved_model_emnist_byclass"
IMAGE_PATH = "test_images/Q.png"

print("Loading model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best.keras'))


def get_character_from_index(idx):
    if idx < 10:
        return str(idx)  # Digits 0-9
    elif idx < 36:
        return chr(ord('A') + idx - 10)  # Uppercase A-Z
    else:
        return chr(ord('a') + idx - 36)  # Lowercase a-z

print(f"Model loaded. Can recognize 62 classes (0-9, A-Z, a-z).")

print(f"\nLoading image: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert('L') 

# Resize to 28x28
img = img.resize((28, 28), Image.LANCZOS)

img_array = np.array(img)

img_array = img_array.astype('float32') / 255.0

img_array = 1.0 - img_array

# img_array = np.transpose(img_array)
img_array = np.fliplr(img_array)

img_array = np.expand_dims(img_array, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

print(f"Image shape: {img_array.shape}")

print("\nMaking prediction...")
predictions = model.predict(img_array, verbose=0)

top_5_idx = np.argsort(predictions[0])[-5:][::-1]

print("\n" + "="*50)
print("PREDICTION RESULTS:")
print("="*50)

for i, idx in enumerate(top_5_idx, 1):
    confidence = predictions[0][idx] * 100
    label = get_character_from_index(idx)
    print(f"{i}. '{label}' - {confidence:.2f}% confidence")

print("="*50)

best_idx = np.argmax(predictions[0])
best_label = get_character_from_index(best_idx)
best_confidence = predictions[0][best_idx] * 100

print(f"\n✓ FINAL PREDICTION: '{best_label}' ({best_confidence:.2f}% confident)")