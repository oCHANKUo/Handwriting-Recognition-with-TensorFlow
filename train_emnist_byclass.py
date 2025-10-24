# train_emnist_byclass.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import os

BATCH_SIZE = 128
EPOCHS = 15  # Increased slightly for better convergence
MODEL_DIR = "saved_model_emnist_byclass"

# 1) Load EMNIST byclass (contains letters+digits)
ds_train, ds_info = tfds.load('emnist/byclass', split='train', with_info=True, as_supervised=True)
ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)

NUM_CLASSES = ds_info.features['label'].num_classes
print(f"Number of classes: {NUM_CLASSES}")
label_names = ds_info.features['label'].names

# Save label names for frontend mapping
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
    json.dump(label_names, f)

# 2) Preprocess function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    
    image = tf.transpose(image, [1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 3) Improved CNN model
def make_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)  # Add dropout earlier
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)  # Added layer
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)  # Increased capacity
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = make_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4) Train
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best.keras'),  # Use .keras format
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 5) Save final SavedModel for TF.js conversion
model.save(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

# Optional: Print final accuracy
final_loss, final_acc = model.evaluate(test_ds)
print(f"Final test accuracy: {final_acc:.4f}")