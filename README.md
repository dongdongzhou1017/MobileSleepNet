
# MobileSleepNet

Code for "MobileSleepNet: A Deployment-Oriented Framework for Sleep Staging on Mobile Devices".
This paper has been submitted to IEEE Journal of Biomedical and Health Informatics (IEEE JBHI). 

## Overview

MobileSleepNet is a lightweight deep learning framework designed for automatic sleep stage classification, optimized for deployment on mobile and edge devices. It utilizes depthwise separable convolutions and model compression techniques to achieve high accuracy with minimal computational resources.


## Features

- Lightweight architecture optimized for mobile/edge deployment
- Depthwise Separable Convolution blocks for efficiency
- Low latency and reduced memory footprint
- Cross-validation evaluation
- Comprehensive evaluation metrics and confusion matrices

## Requirements

- Python 3.7+
- TensorFlow 2.x / PyTorch (depending on implementation)
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn


## Dataset Structure

The project expects preprocessed data in NPZ format with the following structure:
- `Fold_len`: Number of samples in each fold
- `Fold_data`: Preprocessed EEG data segments for each fold
- `Fold_label`: Corresponding sleep stage labels for each fold


## Usage

1. Download the datasets and prepare it in .npz format. Update the data path in the configuration. E.g.
DATA_PATH = "./datasets/sleepedf-2013/npz/sleep_edf_processed_Fpz.npz"
2. Modify the dataset paths and hyperparameters in MobileSleepNet.py
3. Run the training and evaluation:
   ```bash
   python MobileSleepNet.py  # For training and testing the lightweight model
   ```


## Sleep Stages

The model classifies 5 sleep stages:
- W: Wake
- N1: NREM Stage 1
- N2: NREM Stage 2
- N3: NREM Stage 3
- REM: Rapid Eye Movement Sleep


## File Structure

- [MobileSleepNet.py]: Main implementation of the lightweight sleep staging network
- [Utils.py]: Utility functions for evaluation, metrics calculation, and visualization
- [DataGenerator.py]: Data loading, preprocessing, and k-fold cross-validation handling


