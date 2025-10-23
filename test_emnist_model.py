import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "saved_model_emnist_byclass"
MODEL_PATH = os.path.join(MODEL_DIR, "best.h5")
MAPPING_PATH = "emnist-byclass-mapping.txt"  # make sure this file exists

# 1️⃣ Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# 2️⃣ Load EMNIST byclass mapping (index -> ASCII character)
index_to_char = {}
with open(MAPPING_PATH, 'r') as f:
    for line in f:
        idx, ascii_code = line.strip().split()
        index_to_char[int(idx)] = chr(int(ascii_code))

# Optional: check mapping
print("Example: index 10 ->", index_to_char[10])

# 3️⃣ Load user input image
img_path = "test_images/A.jpg"  # replace with your image path
img = tf.keras.utils.load_img(img_path, color_mode="grayscale", target_size=(28,28))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0  # normalize to [0,1]
img_array = 1.0 - img_array
img_array = np.expand_dims(img_array, axis=0)  # shape (1,28,28,1)

# 4️⃣ Predict
pred = model.predict(img_array)

# 5️⃣ Compare against the expected letter
expected_letter = "A"  # the letter user was supposed to draw
target_index = None

for idx, char in index_to_char.items():
    if char.upper() == expected_letter.upper():  # ignore case
        target_index = idx
        break

if target_index is None:
    raise ValueError(f"Expected letter '{expected_letter}' not found in EMNIST mapping!")

# 6️⃣ Calculate confidence / score
confidence = pred[0][target_index]
accuracy_percentage = confidence * 100

print(f"User was supposed to write '{expected_letter}'.")
print(f"Model confidence: {confidence:.2f}")
print(f"Score/Accuracy: {accuracy_percentage:.2f}%")