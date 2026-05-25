import os
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
MODELS_DIR = os.path.join(BASE_DIR, "dataset", "models")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "gesture_model.h5")
LABELS_OUT_PATH = os.path.join(MODELS_DIR, "labels.txt")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

def main():
    print("=== Training CNN Gesture Recognition Model ===")
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Error: {RAW_IMAGES_DIR} not found.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load dataset directly from directories
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        RAW_IMAGES_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        RAW_IMAGES_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Found classes:", class_names)

    # Save labels
    with open(LABELS_OUT_PATH, "w") as f:
        for label in class_names:
            f.write(f"{label}\n")
    print(f"Saved labels to {LABELS_OUT_PATH}")

    # Build CNN Model
    num_classes = len(class_names)
    
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train for a few epochs for demonstration
    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Save Model
    model.save(MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    main()
