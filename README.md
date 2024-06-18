# Prodigy_ML_04

# Hand Gesture Recognition Using Convolutional Neural Networks

This project aims to recognize hand gestures using a Convolutional Neural Network (CNN) built with Keras and TensorFlow. The model is trained on a dataset of hand gesture images and can predict gestures from new images.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Saving the Model](#saving-the-model)
7. [Prediction Function](#prediction-function)
8. [Testing the Model](#testing-the-model)
9. [Model Visualization](#model-visualization)

## Introduction

This project involves creating a CNN to classify hand gestures from images. The model is trained on a dataset stored in Google Drive and is capable of recognizing various hand gestures.

## Setup

### Import Necessary Libraries

Import the essential libraries required for building and training the CNN model, such as Keras, TensorFlow, scikit-image, and others for data manipulation and visualization.

### Mount Google Drive

Mount Google Drive to access the dataset stored in a specific directory.

### Ignore Warnings

Ignore warnings to keep the output clean and focus on essential messages.

### Set Random Seeds for Reproducibility

Set random seeds for NumPy and TensorFlow to ensure the results are reproducible.

### Define Image Size, Batch Size, and Number of Epochs

Define constants for image size, batch size, and the number of training epochs.

## Data Preprocessing

### Preprocess Data

Create a function to preprocess the data using the ImageDataGenerator class from Keras. This function handles the data augmentation and splits the data into training and validation sets.

### Path to the Training Data

Set the path to the directory containing the training data. This path is used to load and preprocess the images.

## Model Architecture

### Create a CNN Model

Define a function to create a CNN model. The model consists of several convolutional layers, max-pooling layers, and dense layers, followed by a softmax activation for multi-class classification.

### Get the Number of Classes

Determine the number of unique classes in the dataset by examining the training data.

### Create the CNN Model

Initialize the CNN model using the number of classes identified.

## Training the Model

### Train the Model

Train the CNN model using the preprocessed training data and validate it on the validation set. Track the training progress over a specified number of epochs.

## Saving the Model

### Save the Model

Save the trained model to a file for later use.

### Print the Model Summary

Print a summary of the model architecture to understand its structure and the number of parameters.

### Plot the Model

Visualize the model architecture by plotting it and saving the plot as an image file.

## Prediction Function

### Load the Saved Model

Load the previously saved model to make predictions on new images.

### Function to Predict the Gesture from an Image

Define a function to preprocess an input image and use the loaded model to predict the hand gesture. The function converts the image to an array, resizes it, normalizes it, and then makes a prediction.

## Testing the Model

### Test the Model on an Image

Use the prediction function to test the model on a sample image and display the predicted gesture.

### Display the Image

Display the test image using matplotlib to visualize the input to the model.

### Test the Model on Another Image

Repeat the prediction process on another sample image to further evaluate the model's performance.

### Display the Image

Display the second test image using matplotlib.

## Model Visualization

### Plot the Model

Visualize the architecture of the model using the `plot_model` function from Keras, showing the layers and their connections. Save this plot as an image file for reference.
