# 🔍 DeepFake Face Detection using EfficientNet-B0 and MobileNetV2

This project explores the ability of deep convolutional neural networks to detect AI-generated faces. We compare two popular architectures—**EfficientNet-B0** and **MobileNetV2**—on a binary classification task using a cleaned subset of the *140K Real and Fake Faces* dataset. Beyond performance, we focus on **interpretability** via Grad-CAM and **representation analysis** through t-SNE.

---

## 🧠 Project Overview

AI-generated faces from GANs and diffusion models are increasingly hard to distinguish from real ones. This project investigates:
- **Model performance** on fake-vs-real classification
- **Grad-CAM** for spatial attention insights
- **t-SNE** for internal feature space separation
- **Bias mitigation** via custom preprocessing

Key takeaway: **EfficientNet-B0** outperformed MobileNetV2 on every metric except inference speed, and it learned more robust, explainable representations.

---

## 🗂️ Repo Contents

| File | Description |
|------|-------------|
| `face_detection_pipeline.ipynb` | End-to-end code: data loading, training, evaluation, Grad-CAM, t-SNE |
| `report.pdf` | Detailed technical report with results, visuals, and discussion |
| `data/` | (Optional) Place to store dataset locally if not loaded dynamically |
| `utils/gradcam.py` | Grad-CAM visualizer (imported in notebook) |
| `utils/tsne_plot.py` | 2D feature visualization via t-SNE |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/deepfake-face-detector.git
cd deepfake-face-detector
```

### 2. Set Up Environment
```bash
pip install -r requirements.txt
# or use your preferred virtual environment manager
```

### 3. Prepare the Dataset

We use a subset of the [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). You can:
- Download it manually from Kaggle
- Place images in `data/real/` and `data/fake/` folders respectively

The notebook handles all preprocessing, resizing, and augmentation.

---

## 📊 Results Summary

| Metric        | EfficientNet-B0 | MobileNetV2 |
|---------------|------------------|--------------|
| Accuracy      | ✅ Higher         | 🔻 Lower     |
| Precision     | ✅ Higher         | 🔻 Lower     |
| Recall        | ✅ Higher         | 🔻 Lower     |
| F1 Score      | ✅ Higher         | 🔻 Lower     |
| Inference Time| 🔻 Slower         | ✅ Faster     |

EfficientNet-B0 consistently made more confident and correct predictions, especially on subtle fake images.

---

## 🧠 Interpretability Tools

### 🎯 Grad-CAM
Used to visualize attention on real vs. fake faces. EfficientNet focused on key facial regions like the eyes and nose bridge, while MobileNet was more diffused.

### 🌐 t-SNE Embeddings
Feature vectors from the penultimate layer of EfficientNet formed two tight, separable clusters—indicating strong internal class distinction.

---

## 📄 Read the Full Report

Check out [`report.pdf`](./report.pdf) for:
- Model architectures
- Training setup
- Grad-CAM visualizations
- t-SNE plots
- Dataset bias correction
- Author reflections

---

## 🧑‍💻 Authors

- **Karan Monga**
- **Akash Deshpande**

---

## 🔗 References

- [Grad-CAM](https://arxiv.org/abs/1610.02391) – Selvaraju et al.
- [t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html) – van der Maaten & Hinton
- [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

---

## 📌 License

MIT License. See `LICENSE` file for details.
