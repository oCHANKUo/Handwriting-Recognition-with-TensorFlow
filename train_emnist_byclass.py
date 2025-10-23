# train_emnist_byclass.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import os

BATCH_SIZE = 128
EPOCHS = 10
MODEL_DIR = "saved_model_emnist_byclass"

# 1) Load EMNIST byclass (contains letters+digits)
ds_train, ds_info = tfds.load('emnist/byclass', split='train', with_info=True, as_supervised=True)
ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)

NUM_CLASSES = ds_info.features['label'].num_classes
print("num classes:", NUM_CLASSES)
label_names = ds_info.features['label'].names  # list of class names (strings)
# Save label names for frontend mapping
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
    json.dump(label_names, f)

# 2) Preprocess function: images are already 28x28 uint8, but tfds provides shape [28,28,1]
def preprocess(image, label):
    # convert to float32 in [0,1], invert if necessary (EMNIST images: white on black?), normalize
    image = tf.cast(image, tf.float32) / 255.0  # shape (28,28,1)
    # Optionally invert so that foreground = 1.0 and background = 0.0:
    # image = 1.0 - image
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 3) Simple CNN model
def make_model(input_shape=(28,28,1), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = make_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4) Train
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best.h5'),
                                       monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)

# 5) Save final SavedModel for TF.js conversion
model.save(MODEL_DIR, save_format='tf')
print("Saved model to", MODEL_DIR)