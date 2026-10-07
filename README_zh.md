# 胸部 X 光片分类（CNN）

> 用 TensorFlow/Keras 搭建的三分类胸片分类器，类别为 **Normal / Covid / Pneumonia**。整个流程（数据增强、带早停的训练、评估、分类报告、混淆矩阵）都在这一个 notebook 里跑完。

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 项目概览

| 项目 | 说明 |
|---|---|
| **任务** | 医学图像多分类（3 类） |
| **方法** | CNN（3 个卷积块 + 全连接头）+ 图像增强 |
| **框架** | TensorFlow / Keras |
| **类别** | `Normal` / `Covid` / `Pneumonia` |
| **输入** | 224×224×3 RGB 图像 |
| **开源协议** | MIT |

## 为什么做这个项目

医学图像分类是最经典的"第一个真正 CNN"任务，也是练全流程的好载体：数据加载、用增强对抗过拟合、用回调（早停）、以及——很关键——学会看**混淆矩阵**而不是只盯着准确率。三分类设定（Normal / Covid / Pneumonia）还会逼你想清楚类别不平衡和每类的召回率。

## 网络结构

```
输入 224×224×3
  ├─ Conv2D(32, 3×3, relu) → MaxPool(2×2)
  ├─ Conv2D(64, 3×3, relu) → MaxPool(2×2)
  ├─ Conv2D(128, 3×3, relu) → MaxPool(2×2)
  ├─ Flatten
  ├─ Dense(256, relu) → Dropout(0.5)
  └─ Dense(3, softmax)
优化器: Adam · 损失: categorical_crossentropy · 指标: accuracy
EarlyStopping(monitor=val_accuracy, patience=5, restore_best_weights=True)
```

训练时对训练集用 `ImageDataGenerator` 增强（旋转/平移/剪切/缩放/翻转），并切出 20% 作为验证集；测试集**不做增强**以保证评估公平。

## 数据集

图片数据**不随本仓库提供**（体积过大，不适合 Git）。请使用任何带 Normal / Covid / Pneumonia 标签的胸片数据集——例如 Kaggle 上的 *COVID-19 Radiography Database*——并按如下结构放置：

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

notebook 通过 `flow_from_directory` 直接读取 `dataset/train_dataset` 与 `dataset/test_dataset`，类别固定为 `['Normal','Covid','Pneumonia']`。

## 目录结构

```
chest-xray-cnn/
├── chest_xray_cnn.ipynb     # 完整流程（仅代码；运行即可复现输出）
├── dataset/                 # 不在仓库内——请自行添加（见上）
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
# 1. 按上面结构把数据放到 dataset/ 下
# 2. 打开 notebook
jupyter notebook chest_xray_cnn.ipynb
# 运行全部单元格；末尾会打印测试准确率、训练曲线、分类报告和混淆矩阵
```

## 效果

| 模型 | 测试准确率 |
|---|---|
| 3 层卷积 CNN（本仓库） | **TODO —— 跑完填入**（简历标注约 95.65%；请用你的数据划分实测验证） |

> ⚠️ 简历里写的 95.65% 来自某个特定数据划分，**尚未在本仓库环境中复现验证**。请运行 notebook，把真实测得的数字填进 TODO 后再在任何地方引用。

## 诚实的局限性

- `plt.rcParams['font.family'] = ['SimHei']` 依赖 **SimHei（黑体）** 字体，仅 Windows 自带。在 Linux/macOS/Colab 上需安装中文字体，或删掉这一行（图例会回退为英文）。
- 只在单一固定划分上训练，未做 k 折交叉验证。
- 没有用预训练骨干网络——ResNet 迁移学习基线大概率能超过这个从零训练的 CNN。

## 接下来想做的

- [ ] 换上 ResNet50 / EfficientNet 骨干并对比。
- [ ] 加 Grad-CAM，展示模型关注肺部的哪些区域。
- [ ] 导出训练好的模型，用 Gradio 包成可交互 demo。

---

<sub>由 <a href="https://github.com/lannawhite">@lannawhite</a> 构建 · 北部湾大学 人工智能专业 本科生</sub>
