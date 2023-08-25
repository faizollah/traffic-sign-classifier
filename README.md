# Traffic Sign Image Classifier Using the Sequential API

This repository contains a Jupyter notebook that demonstrates how to build an image classifier using TensorFlow's Sequential API. The classifier is trained to recognize traffic signs from the GTSRB - German Traffic Sign Recognition Benchmark dataset.

## Overview
The notebook provides a step-by-step guide to:

- Load the traffic sign dataset from Google Drive.
- Preprocess the images by resizing and normalizing them.
- Build a neural network model using the Sequential API with dropout layers to prevent overfitting.
- Train the model using early stopping to monitor validation loss.
- Visualize the training and validation loss and accuracy.
- Save, load, and convert the trained model to TensorFlow Lite (TFLite) format for deployment on mobile or embedded devices.

## Requirements
- TensorFlow 2.x
- PIL (Python Imaging Library)
- NumPy
- scikit-learn
- matplotlib
- pandas
- Google Colab (for mounting Google Drive and accessing the dataset)

## Dataset
The dataset used is the [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train). It contains thousands of images of 43 different traffic signs.

## Model Architecture
The model is built using TensorFlow's Sequential API. It consists of:

- An input layer that flattens the images.
- Three dense layers with ReLU activation.
- Dropout layers to prevent overfitting.
- An output layer with softmax activation for multi-class classification.

## Visualization
The notebook provides various visualizations to monitor the training process, including:

- Line plots for training and validation loss and accuracy.
- Histograms of training loss and accuracy.
- A combined plot of loss and accuracy over epochs.

## Saving and Conversion
After training, the model is saved in the HDF5 format. It is then loaded and converted to the TensorFlow Lite (TFLite) format, which is suitable for deployment on mobile or embedded devices.
