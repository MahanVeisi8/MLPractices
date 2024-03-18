# README - Practice Number 2: Stroke Prediction and Insurance Cost Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your_drive_link_here)
[![Python Versions](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/)
[![Dependencies Status](https://img.shields.io/badge/Dependencies-up%20to%20date-brightgreen)](https://github.com/your_username/repository/blob/main/requirements.txt)

## Introduction

Welcome to Practice Number 2! In this practice, we explore two important machine learning tasks:

- **Stroke Prediction**
- **Insurance Cost Prediction**

We delve into preprocessing steps, model building, and evaluation techniques to achieve accurate predictions.

## Table of Contents

***Stroke Prediction***
- [Data Preprocessing](#data-preprocessing)
  - [Data Loading and Cleaning](#data-loading-and-cleaning)
  - [Handling Missing Values](#handling-missing-values)
  - [Encoding Categorical Variables](#encoding-categorical-variables)
  - [Train-Test Split](#train-test-split)
- [Model Building](#model-building)
  - [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
  - [Results and Analysis](#results-and-analysis)

***Insurance Cost Prediction***
- [Data Preprocessing](#data-preprocessing-1)
  - [Data Loading and Exploration](#data-loading-and-exploration)
  - [Handling Missing Values](#handling-missing-values-1)
  - [Feature Scaling](#feature-scaling)
  - [Train-Test Split](#train-test-split-1)
- [Model Building](#model-building-1)
  - [Linear Regression](#linear-regression)
  - [Polynomial Regression](#polynomial-regression)
  - [Results and Analysis](#results-and-analysis-1)

# Stroke Prediction

## Data Preprocessing

### Data Loading and Cleaning

- The dataset containing information about individuals and their stroke history is loaded.
- Irrelevant features such as 'id' are dropped from the dataset.

### Handling Missing Values

- Missing values in numerical features are imputed using the mean.
- Categorical features are imputed using the most frequent value.

### Encoding Categorical Variables

- Categorical variables are one-hot encoded for model training.

### Train-Test Split

- The dataset is split into training and testing sets for model evaluation.

## Model Building

### Support Vector Classifier (SVC)

- A linear SVC model is trained on the preprocessed data.
- Model performance is evaluated using accuracy, confusion matrix, and classification report metrics.

### Results and Analysis

- Despite preprocessing efforts, the initial SVC model performs poorly.
- Further investigation reveals issues with data quality and class imbalance.
- A new dataset is created by undersampling the majority class, leading to improved model performance.
- Custom threshold adjustment techniques are applied to enhance model performance further.

# Insurance Cost Prediction

## Data Preprocessing

### Data Loading and Exploration

- The insurance dataset is loaded and explored to understand its structure.

### Handling Missing Values

- Missing values are handled, and categorical variables are one-hot encoded.

### Feature Scaling

- Numerical features are scaled using standard scaling to ensure uniformity.

### Train-Test Split

- The dataset is split into training and testing sets for model training.

## Model Building

### Linear Regression

- A simple linear regression model is built to predict insurance costs.

### Polynomial Regression

- Polynomial features are added to capture non-linear relationships, improving model performance.

### Results and Analysis

- The linear regression model provides decent accuracy in predicting insurance costs.
- Polynomial regression outperforms the linear model, even with fewer training epochs.
- Model predictions are visualized against actual insurance costs, demonstrating their effectiveness.

## Conclusion

In conclusion, this practice showcases the application of machine learning techniques in predicting strokes and insurance costs. Through data preprocessing, model training, and evaluation, we achieve meaningful insights and predictive accuracy in both tasks, highlighting the importance of careful data handling and model selection.

