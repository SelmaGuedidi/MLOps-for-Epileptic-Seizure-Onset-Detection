# Epilepsy Onset Detection using MLOps

## Overview
This repository contains code for developing a machine learning model to detect the onset of epileptic seizures. The model is built using MLOps best practices to ensure efficient development, deployment, and management.

## Dataset
The dataset used for this project consists of physiological signal data collected from patients with epilepsy. The signals were preprocessed and merged into a single CSV file for ease of use.

## Features
The features extracted from the physiological signals include:
- Detrended Fluctuation Analysis (DFA)
- Fisher Information
- Higuchi Fractal Dimension (HFD)
- Petrosian Fractal Dimension (PFD)
- Singular Value Decomposition (SVD) Entropy
- Variance
- Standard Deviation
- Mean
- Variance of Fast Fourier Transform (FFT)
- Standard Deviation of FFT
- Variance of Second FFT (FFT2)
- Zero Crossing Rate
- Complexity

## Models
Several machine learning models were trained and evaluated for seizure onset detection, including:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Gradient Boosting
- AdaBoost

Each model was also trained using principal component analysis (PCA) for dimensionality reduction.

## Pipeline
The pipeline for model development and evaluation includes the following steps:
1. Data preprocessing and merging CSV files.
2. Splitting the dataset into training and validation sets.
3. Building machine learning pipelines using MLOps tools.
4. Training and hyperparameter tuning using cross-validation.
5. Evaluating model performance using accuracy, recall, and F1-score.
6. Exporting the trained models for deployment.

## Results
The best-performing model achieved an accuracy of 96.30% on the test set, with corresponding recall and F1-score values of 96.30% and 96.06%, respectively. The models were fine-tuned using hyperparameter optimization to achieve these results.

## Getting Started
To get started with this project, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Start the MLFlow server by running the following command:
`mlflow server --host 127.0.0.1 --port 8080`
4. Run the pipeline.py script to initiate the MLFlow pipeline.
5. Experiment with different models and hyperparameters as needed.
