# Chest X-Ray Classification (CNN)

> A 3-class chest X-ray classifier built with TensorFlow/Keras — **Normal / Covid / Pneumonia**. The whole pipeline (data augmentation, training with early stopping, evaluation, classification report, confusion matrix) runs in one notebook.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

| Item | Detail |
|---|---|
| **Task** | Multi-class medical image classification (3 classes) |
| **Approach** | CNN (3 conv blocks + dense head) with image augmentation |
| **Framework** | TensorFlow / Keras |
| **Classes** | `Normal` / `Covid` / `Pneumonia` |
| **Input** | 224×224×3 RGB images |
| **License** | MIT |

## Why this project

Medical-image classification is the classic "first real CNN" task and a good vehicle for learning the full loop: data loading, augmentation to fight overfitting, callbacks (early stopping), and — importantly — how to read a **confusion matrix** instead of just staring at accuracy. The three-class setup (Normal / Covid / Pneumonia) also forces you to think about class imbalance and per-class recall.

## Architecture

```
input 224×224×3
  ├─ Conv2D(32, 3×3, relu) → MaxPool(2×2)
  ├─ Conv2D(64, 3×3, relu) → MaxPool(2×2)
  ├─ Conv2D(128, 3×3, relu) → MaxPool(2×2)
  ├─ Flatten
  ├─ Dense(256, relu) → Dropout(0.5)
  └─ Dense(3, softmax)
optimizer: Adam · loss: categorical_crossentropy · metric: accuracy
EarlyStopping(monitor=val_accuracy, patience=5, restore_best_weights=True)
```

Training uses `ImageDataGenerator` augmentation (rotation / shift / shear / zoom / flip) on the training split, with a 20% `validation_split` held out; the test set is evaluated **without** augmentation for a fair score.

## Dataset

The image data is **not included in this repo** (too large for Git). Use any chest X-ray set that provides Normal / Covid / Pneumonia labels — for example the *COVID-19 Radiography Database* on Kaggle — and arrange it as:

```
dataset/
├── train_dataset/
│   ├── Normal/
│   ├── Covid/
│   └── Pneumonia/
└── test_dataset/
    ├── Normal/
    ├── Covid/
    └── Pneumonia/
```

The notebook reads `dataset/train_dataset` and `dataset/test_dataset` directly via `flow_from_directory`, with `classes=['Normal','Covid','Pneumonia']`.

## Project structure

```
chest-xray-cnn/
├── chest_xray_cnn.ipynb     # the full pipeline (exploratory notebook)
├── model.py                 # build_model(): CNN definition
├── train.py                 # end-to-end training -> models/best_model.keras + results/*.png
├── predict.py               # single-image inference with confidence
├── dataset/                 # NOT in repo — add your own (see Dataset)
├── models/                  # NOT in repo — trained weights (gitignored)
├── results/                 # NOT in repo — plots (gitignored)
├── requirements.txt
└── README.md
```

## Quick start

**Option A — runnable scripts (recommended):**

```bash
pip install -r requirements.txt
# 1. put your data under dataset/ as shown above
# 2. train (saves models/best_model.keras and results/*.png)
python train.py --data_root dataset
# 3. infer a single image
python predict.py --image path/to/xray.jpg
```

**Option B — the notebook (for exploration):**

```bash
jupyter notebook chest_xray_cnn.ipynb
# run all cells; the final cells print test accuracy, a training curve,
# a classification report and a confusion matrix
```

> `train.py` / `predict.py` use **English** plot labels and avoid the Windows-only SimHei font, so they run cross-platform. The exploratory notebook keeps the original Chinese labels.

## Results

| Model | Test accuracy |
|---|---|
| 3-block CNN (this repo) | **TODO — run and fill in** (résumé states ~95.65%; verify on your data split) |

> ⚠️ The 95.65% figure quoted in the résumé comes from a specific data split and has **not been re-verified in this repo's environment**. Run the notebook and replace the TODO with your measured number before citing it anywhere.

## Honest limitations

- `plt.rcParams['font.family'] = ['SimHei']` assumes the **SimHei** font, which is Windows-only. On Linux/macOS/Colab, either install a CJK font or delete that line (plots fall back to English labels).
- Trained on a single fixed split; no k-fold cross-validation is reported.
- No pre-trained backbone — a ResNet transfer-learning baseline would likely beat this from-scratch CNN.

## What I'd do next

- [ ] Swap in a ResNet50 / EfficientNet backbone and compare.
- [ ] Add Grad-CAM to show which lung regions drive each prediction.
- [ ] Export the trained model and wrap it in a small Gradio demo.

---

<sub>Built by <a href="https://github.com/lannawhite">@lannawhite</a> · AI undergraduate, Beibu Gulf University</sub>
