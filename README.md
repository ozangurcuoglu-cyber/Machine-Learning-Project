# Machine Learning Homeworks

This repository contains my university machine learning assignments.

## Contents
- **PNN Models Assignment** — Implementation and comparison of Probabilistic Neural Network architectures.
- **Predicting Road Accidents (Kaggle Competition)** — End-to-end data pipeline, feature engineering, and model training for road accident prediction.

---

**Author:** Ozan Gürcüoğlu  
**Environment:** Python, Jupyter Notebook, scikit-learn, NumPy, pandas







# Predicting Road Accidents (Kaggle Competition) --> **ML_HW_3.ipynb**

A Kaggle competition project focused on predicting the likelihood of road accidents based on environmental and traffic-related data.

## Key Steps
- Data cleaning and feature engineering
- Model selection and hyperparameter tuning
- Evaluation with ROC-AUC and F1 metrics
- Submission to Kaggle for leaderboard evaluation




# PNN Models Assignment-- --> **HW_5.ipynb**

This notebook explores Probabilistic Neural Networks (PNN) and compares their performance with other supervised learning models.

## Highlights
- Implemented PNN using NumPy and scikit-learn.
- Compared accuracy and computational efficiency against traditional feedforward networks.
- Evaluated model performance on a benchmark dataset.



# Math482 – Assignment 4: Derivation of Loss Functions

In this assignment, the objective is to **derive loss functions** starting from the probability distribution of a dataset and understand how different artificial intelligence models are trained under various distributional assumptions.

## 📘 Purpose
The main goal is to learn how to mathematically derive a loss function using the **Gaussian (normal) distribution**, and to connect this derivation to commonly used functions in machine learning such as **MSE (Mean Squared Error)**.

Specifically, this notebook includes:
- Derivation of the loss function directly from the **negative log-likelihood** of the Gaussian distribution.  
- Implementation and comparison of its simplified form, the **MSE loss function**.  
- Integration of the derived loss into an existing neural network structure (activation, forward/backward propagation, training functions).

## 🧠 Learning Outcomes
By completing this notebook, you will:
- Understand the relationship between probability theory and optimization in neural networks.  
- See how assumptions on data distribution influence the form of the loss function.  
- Implement and analyze loss derivations in a Jupyter Notebook environment.

## ⚙️ Environment
- **Language:** Python  
- **Libraries:** NumPy, Matplotlib  
- **Platform:** Jupyter Notebook

---

**Author:** Ozan Gürcüoğlu  
**Course:** Math482 – Machine Learning Theory  
**Date:** November 2025




# 🩺 Diabetes Prediction Project

Bu proje, hastaların tıbbi ölçümlerini kullanarak diyabet riskini tahmin etmek amacıyla geliştirilmiş bir **Makine Öğrenmesi** çalışmasıdır. Veri setindeki çeşitli sağlık parametreleri analiz edilerek, bir kişinin diyabet hastası olup olmadığı yüksek doğruluk oranıyla öngörülmeye çalışılmıştır.

## 🚀 Proje Özeti
Diyabet, dünya genelinde milyonlarca insanı etkileyen kronik bir hastalıktır. Erken teşhis, hastalığın yönetimi için kritiktir. Bu proje; veri temizleme, özellik mühendisliği (feature engineering) ve sınıflandırma algoritmalarını kullanarak sağlık verilerinden anlamlı sonuçlar çıkarmayı hedefler.



## 🛠️ Kullanılan Teknolojiler & Kütüphaneler
* **Dil:** Python 3.x
* **Veri Analizi:** Pandas, NumPy
* **Görselleştirme:** Matplotlib, Seaborn
* **Makine Öğrenmesi:** Scikit-learn (Logistic Regression, Random Forest, SVM vb.)
* **Model Kaydetme:** Pickle / Joblib

## 📊 Veri Seti Hakkında
Projede (örneğin: Pima Indians Diabetes Dataset) kullanılmıştır. Temel özellikler şunlardır:
* **Pregnancies:** Gebelik sayısı
* **Glucose:** Glikoz değeri
* **Blood Pressure:** Kan basıncı
* **BMI:** Vücut kitle indeksi
* **Age:** Yaş
* **Outcome:** Diyabet durumu (0: Negatif, 1: Pozitif)

## 🏗️ İş Akışı (Workflow)
1. **Veri Ön İşleme:** Eksik değerlerin (0 olan mantıksız veriler) analizi ve doldurulması.
2. **EDA (Keşifçi Veri Analizi):** Korelasyon matrisleri ve dağılım grafiklerinin incelenmesi.
3. **Özellik Ölçeklendirme:** StandardScaler veya MinMaxScaler kullanımı.
4. **Model Eğitimi:** Farklı algoritmaların (Random Forest, XGBoost vb.) karşılaştırılması.
5. **Değerlendirme:** Confusion Matrix, F1-Score ve Accuracy değerlerinin analizi.

## 📈 Sonuçlar
Modelimiz test verileri üzerinde şu başarı metriklerini elde etmiştir:
* **Accuracy:** %XX
* **Precision:** %XX
* **Recall:** %XX

## 💻 Kurulum ve Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/kullaniciadi/diabetes-prediction.git](https://github.com/kullaniciadi/diabetes-prediction.git)
