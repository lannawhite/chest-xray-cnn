"""Model definition for the 3-class chest X-ray CNN classifier.

Classes: Normal / Covid / Pneumonia.
Input: RGB images resized to 224x224, pixel values rescaled to [0, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers


VALID_CLASS_NAMES = ["Normal", "Covid", "Pneumonia"]
IMG_SIZE = (224, 224)
NUM_CLASSES = len(VALID_CLASS_NAMES)


def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """Build the sequential CNN used in this project.

    Architecture: 3 conv blocks (32 -> 64 -> 128) with max-pooling,
    followed by a 256-unit Dense head with Dropout(0.5) and a softmax output.
    """
    model = tf.keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
