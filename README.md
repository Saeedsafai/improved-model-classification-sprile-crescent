# Improved Model Classification – Spirals & Crescentsv Datasets

This repository contains an AI project comparing a **base model** and an **improved model** on two benchmark datasets: **Crescentsv** and **Spirals**.  
The improved model achieves significantly higher accuracy and demonstrates the effect of better feature representation and optimization.

---

## 🧠 Project Overview
The goal of this project is to analyze and enhance the performance of classification models on complex synthetic datasets.  
Two versions of the model were trained and evaluated:
- **Base Model:** A simple classifier with minimal tuning.
- **Improved Model:** Enhanced architecture with optimization, normalization, and hyperparameter tuning.

---

## 📊 Results Summary

| Model | Dataset | Validation Accuracy | Train Accuracy | Test Accuracy | Mean | Std |
|--------|----------|--------------------|----------------|----------------|------|------|
| Improved Model | crescentsv | **100.0%** | **99.92%** | **100.0%** | 99.92% | 1.11e−16 |
| Improved Model | spirals | **100.0%** | **100.0%** | **100.0%** | 100.0% | 0 |
| Base Model | crescentsv | 81.75% | 80.2% | 84.5% | 80.2% | 0 |
| Base Model | spirals | 52.5% | 61.1% | 56.0% | 61.1% | 1.11e−16 |

---

## ⚙️ Technologies Used
- **Python 3**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- (optional) **TensorFlow / Keras** if deep learning was used

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Saeedsafai/improved-model-classification-sprile-crescent.git
   cd improved-model-classification-sprile-crescent
