# Meat Classification using CNN


## Introduction

This projects aims to classify Meat images using Convolutional Neural Networks (CNN). The dataset used is the [Meat Freshness Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset). 

## Dataset
This dataset contains 2266 images of meat in three different condition.
1. Fresh
2. Spoiled
3. Half Fresh

The dataset is divided into 2 folders train and test. The train folder contains 1813 images and the test folder contains 453 images. The images are in jpg format. All the images are of the same size 416x416x3.  

## Data Preprocessing

The names of the images denoted their class.
1. Fresh images are denoted by the prefix 'FRESH'
2. Spoiled images are denoted by the prefix 'SPOIL'
3. Half Fresh images are denoted by the prefix 'HALF'

The images were stored into their respective arrays and the labels were encoded into 0,1 and 2 for Fresh, Spoiled and Half Fresh respectively. The images were then converted into numpy arrays and normalized.

## Model

Model was as Follows:
```python
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(416,416,3)),
    MaxPooling2D((2,2)),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout((0.4)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout((0.4)),
    Dense(3, activation='softmax')
    
])
```

## Training

The model was trained for 20 epochs. The model was compiled using Adam optimizer and categorical crossentropy Loss Functions. The mode was trained on Kaggle Notebook and achieve an accuracy of 0.96 on the training set and 0.92 on the validation set.


