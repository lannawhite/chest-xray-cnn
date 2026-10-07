"""Single-image inference for the chest X-ray CNN.

Usage:
    python predict.py --image path/to/xray.jpg
    python predict.py --image x.jpg --model models/best_model.keras

Prints the predicted class and the confidence (softmax probability).
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from model import build_model, VALID_CLASS_NAMES, IMG_SIZE


def load_model_safe(model_path: str):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    raise FileNotFoundError(
        f"Model not found at {model_path}. Run `python train.py` first "
        "or pass --model with a trained .keras file."
    )


def predict_image(model, img_path: str) -> tuple[str, float, dict]:
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return VALID_CLASS_NAMES[idx], float(probs[idx]), dict(zip(VALID_CLASS_NAMES, probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single chest X-ray")
    parser.add_argument("--image", required=True, help="Path to an X-ray image")
    parser.add_argument(
        "--model", default="models/best_model.keras", help="Trained .keras model"
    )
    args = parser.parse_args()

    model = load_model_safe(args.model)
    label, conf, all_probs = predict_image(model, args.image)

    print(f"Image : {args.image}")
    print(f"Result: {label}  (confidence {conf:.2%})")
    print("Per-class probabilities:")
    for name, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s}: {p:.2%}")


if __name__ == "__main__":
    main()
