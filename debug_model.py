import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

MODEL_DIR = "saved_model_emnist_byclass"
IMAGE_PATH = "test_images/b.png"  # your b image

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best.keras'))

img = Image.open(IMAGE_PATH).convert('L')
img_resized = img.resize((28, 28), Image.LANCZOS)

img_array = np.array(img_resized).astype('float32') / 255.0
img_inverted = 1.0 - img_array

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(img_inverted, cmap='gray')
axes[1].set_title('Inverted')
axes[1].axis('off')

axes[2].imshow(img_inverted, cmap='gray')
axes[2].set_title('Final (What model sees)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_b.png', dpi=150)
print("Saved to debug_b.png")