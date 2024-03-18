# Practice Number 1 - README

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uR806lqfcw2c1COt5zGLLJ2o6E6umLXy?usp=sharing)
[![Python Versions](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/)
[![Dependencies Status](https://img.shields.io/badge/Dependencies-up%20to%20date-brightgreen)](https://github.com/username/repository/blob/main/requirements.txt)



## Introduction
In this practice, we explore two fundamental machine learning algorithms: 
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**

We implement these algorithms, evaluate their performance, and explore techniques for hyperparameter tuning to improve model accuracy.

## Table of Contents
- [Decision Trees Implementation](#section-a-implementation-of-decision-trees)
- [Model Evaluation and Hyperparameter Tuning](#section-b-model-evaluation-and-hyperparameter-tuning)
- [Random Forest Implementation and Evaluation](#section-c-random-forest-implementation-and-evaluation)
- [Gradient Boosting Implementation and Evaluation](#section-d-gradient-boosting-implementation-and-evaluation)
- - [Contribution Guidelines](#contribution-guidelines)


# Decision Trees and Random Forest Classifier
Decision Trees Practice

This repository contains code for implementing and evaluating Decision Trees for multi-class classification tasks. The practice is divided into several sections:

### Section A: Implementation of Decision Trees
In this section, we implement a complete Decision Tree classifier from scratch using Python and NumPy. The implementation includes the following components:

- Node Class: Defines the structure of a node in the decision tree.
- DecisionTree Class: Represents the Decision Tree model with methods for fitting and predicting.
- Helper Functions: Includes functions for calculating entropy, information gain, finding the best split, and building the decision tree recursively.

### Section B: Model Evaluation and Hyperparameter Tuning

- ***Model Training and Evaluation:*** We trained several Decision Tree classifiers using different hyperparameters and evaluated their performance on a given dataset. The hyperparameters explored include:
- - Split criterion (Gini impurity or Entropy)
- - Maximum depth of the tree
- - Minimum samples required to split a node
- - Minimum samples required to be at a leaf node
We used the DecisionTreeClassifier from scikit-learn to create and train the models, and then measured their accuracy on the training data.

- ***Hyperparameter Tuning:***
 To find the optimal hyperparameters, we performed a grid search over a range of values for max_depth and min_samples_leaf, using a validation set. We visualized the learning curves to understand the model's performance as the training set size increases.

### Section C: Random Forest Implementation and Evaluation
In this section, we implemented a Random Forest classifier using scikit-learn's RandomForestClassifier. We trained the Random Forest with 100 trees and evaluated its performance on a validation set.

### Section D: Gradient Boosting Implementation and Evaluation
We implemented a Gradient Boosting classifier using scikit-learn's GradientBoostingClassifier. We tuned hyperparameters such as max_depth and learning_rate using GridSearchCV to find the best combination for our model. We evaluated the final model's performance on a test set and visualized the results using a confusion matrix.


## Contribution Guidelines
We welcome contributions from the community! If you encounter any bugs or have suggestions for improvements, please open an issue on GitHub. If you'd like to contribute code, feel free to fork the repository and submit a pull request with your changes.
