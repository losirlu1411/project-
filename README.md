# project-Tuberculosis Detection using CNNs

This project aims to develop and evaluate machine learning models, including a custom CNN and pretrained models (VGG16, DenseNet201, ResNet101), for automated detection of tuberculosis from chest X-ray images. Below is an explanation of the codebase.


This is a data set link to access the data https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset.

Table of Contents

Introduction

Prerequisites

Dataset Description

Code Overview

Imports and Configuration

Data Loading and Preprocessing

Class Weights and Data Augmentation

Model Building

Training and Evaluation Framework

Model Training

Visualization of Results

Usage Instructions

Results and Insights

Introduction

This project uses convolutional neural networks (CNNs) for tuberculosis detection from chest X-ray images. It compares the performance of a custom CNN model with popular pretrained models (VGG16, DenseNet201, ResNet101) on both the full dataset and a sampled subset.

Prerequisites

Ensure the following libraries are installed:

Python 3.x

TensorFlow

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

OpenCV

Additionally, store the dataset in Google Drive or locally and update the base_path in the code to reflect its location.

Dataset Description

The dataset includes X-ray images divided into two categories:

Normal

Tuberculosis

Images are resized to 224x224 for uniformity. The dataset is split into training (80%) and testing (20%) subsets. Class weights and data augmentation are applied to handle class imbalance and improve model robustness.

Code Overview

1. Imports and Configuration

The necessary libraries for data processing, model building, and evaluation are imported. Key configurations include the dataset path (base_path), image size (224x224), and class labels (Normal, Tuberculosis).

2. Data Loading and Preprocessing

Function: load_data(base_path, labels, image_size, sample_size)

Purpose: Loads and preprocesses images by resizing them and normalizing pixel values.

Output: Returns preprocessed images (x_data) and one-hot encoded labels (y_data).

3. Class Weights and Data Augmentation

Class Weights: Computes weights to balance the dataset for fair training.

Data Augmentation: Introduces transformations like rotation, zoom, and flipping to increase training data diversity.

4. Model Building

Custom CNN Model:

Architecture:

Three convolutional layers with increasing filters (32, 64, 128).

Max pooling for down-sampling.

Dense layer (256 neurons) with dropout for regularization.

Output layer with softmax activation.

Compilation:

Optimizer: Adam

Loss: Categorical cross-entropy with label smoothing (0.1)

Metrics: Accuracy and recall.

Pretrained Models:

Base Models: VGG16, DenseNet201, ResNet101 (pretrained on ImageNet).

Fine-tuned by freezing base layers and adding:

Flatten layer.

Dense layer (256 neurons).

Dropout (50%).

Softmax output layer.

5. Training and Evaluation Framework

Function: train_and_evaluate_model()

Purpose: Trains the model and evaluates it using the test set.

Key Features:

Early stopping to prevent overfitting.

Classification metrics: Accuracy, precision, recall, and F1-score.

Visualization of confusion matrix using a heatmap.

6. Model Training

Models are trained on both the full dataset and a sampled subset (500 images per class):

Custom CNN: Constructed and trained.

Pretrained Models: VGG16, DenseNet201, ResNet101 fine-tuned for tuberculosis classification.

7. Visualization of Results

Plots:

Training vs Validation Accuracy.

Training vs Validation Loss.

Prediction Display:

Displays random test images with true and predicted labels. Correct predictions are highlighted in green; incorrect ones in red.

Usage Instructions

Clone or download this repository.

Place the tuberculosis dataset in the specified path (base_path).

Run the code in a Python environment or Jupyter Notebook.

View the results and performance comparison in the generated plots and metrics.

Results and Insights

The models are evaluated on metrics such as accuracy, precision, recall, and F1-score. Bar plots provide a comparative analysis of model performance on the full dataset and the sampled dataset. Insights include:

DenseNet201 outperformed other models in both datasets.

The Custom CNN achieved competitive performance with lower computational requirements.

ResNet101 struggled in the sampled dataset due to its depth and reliance on large datasets for effective training.
