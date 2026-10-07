"""End-to-end training script for the chest X-ray 3-class CNN.

Usage:
    python train.py --data_root dataset

Expected layout under data_root:
    dataset/
        train_dataset/{Normal,Covid,Pneumonia}/...jpg
        test_dataset/ {Normal,Covid,Pneumonia}/...jpg

Outputs:
    models/best_model.keras   best checkpoint (by val_accuracy)
    results/training_history.png
    results/confusion_matrix.png
    prints classification_report on the test set

Note: plots use English labels so the script runs cross-platform without
the Windows-only SimHei font.
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / CI friendly
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_model, VALID_CLASS_NAMES, IMG_SIZE, NUM_CLASSES

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_generators(data_root: str, img_size, batch_size: int):
    train_dir = os.path.join(data_root, "train_dataset")
    test_dir = os.path.join(data_root, "test_dataset")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        classes=VALID_CLASS_NAMES,
    )
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        classes=VALID_CLASS_NAMES,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=VALID_CLASS_NAMES,
    )
    return train_generator, val_generator, test_generator


def plot_history(history, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc", color="blue")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train / Val Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", color="blue")
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train / Val Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred_classes, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(NUM_CLASSES))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=VALID_CLASS_NAMES,
        yticklabels=VALID_CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chest X-ray CNN")
    parser.add_argument(
        "--data_root",
        default="dataset",
        help="Root dir containing train_dataset/ and test_dataset/ subdirs",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_gen, val_gen, test_gen = build_generators(
        args.data_root, IMG_SIZE, args.batch_size
    )

    model = build_model(NUM_CLASSES)

    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    )
    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[early_stopping, checkpoint],
    )

    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_history(history, "results/training_history.png")

    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    print(
        classification_report(
            y_true,
            y_pred_classes,
            labels=np.arange(NUM_CLASSES),
            target_names=VALID_CLASS_NAMES,
        )
    )
    plot_confusion(y_true, y_pred_classes, "results/confusion_matrix.png")
    print("Saved artifacts: models/best_model.keras, results/*.png")


if __name__ == "__main__":
    main()
