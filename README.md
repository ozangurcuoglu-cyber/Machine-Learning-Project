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




# 🩺 Diabetes Diagnosis Prediction (Kaggle S5E12)

This project focuses on predicting diabetes diagnosis using a large-scale dataset from the **Kaggle Playground Series (Season 5, Episode 12)**. The goal is to build a classification model that accurately identifies whether a patient has diabetes based on various health indicators, lifestyle habits, and clinical measurements.

## 📊 Dataset Overview
The dataset contains **700,000 patient records** with 26 distinct features, providing a rich ground for deep statistical analysis and machine learning.

**Key Features include:**
* **Clinical Data:** BMI, Blood Pressure (Systolic/Diastolic), Cholesterol Levels (HDL/LDL), Triglycerides, Heart Rate.
* **Lifestyle Habits:** Alcohol consumption, physical activity, diet score, sleep hours, screen time, and smoking status.
* **Demographics:** Age, Gender, Ethnicity, Education, and Income levels.
* **Medical History:** Family history of diabetes, hypertension, and cardiovascular history.
* **Target:** `diagnosed_diabetes` (Classification)

## 🔍 Exploratory Data Analysis (EDA)
In the notebook, I conducted thorough data profiling:
* **Target Distribution:** Visualized the balance of the `diagnosed_diabetes` target variable using Seaborn countplots.
* **Correlation Profiling:** Generated a high-resolution **Correlation Heatmap** to identify the strongest predictors of diabetes and understand the relationships between clinical variables (e.g., BMI vs. Blood Pressure).
* **Statistical Analysis:** Descriptive statistics (`df.describe()`) to understand outliers and feature scaling requirements.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Kaggle Notebooks (GPU Enabled - Tesla T4)

## 🚀 Project Workflow
1. **Data Loading:** Efficiently handling 700k rows using Pandas.
2. **Preprocessing:** Analyzing feature types (int64, float64, object) and checking for missing values.
3. **Statistical Summary:** Understanding the distribution of clinical metrics.
4. **Feature Analysis:** Evaluating correlations to filter significant features for the model.

## 📈 Key Insights
* The correlation matrix revealed significant patterns between age, weight-related metrics (BMI, Waist-to-Hip ratio), and the likelihood of diagnosis.
* Lifestyle factors like physical activity and diet scores show measurable impacts on health markers.

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/diabetes-prediction-kaggle.git](https://github.com/yourusername/diabetes-prediction-kaggle.git)
