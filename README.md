# Braille Character Recognition using Deep Learning

## Overview

This project focuses on building a computer vision system that can recognize Braille characters from images and convert them into corresponding English alphabets. The goal was to take raw image data, process it properly, train a deep learning model, and finally deploy it in the form of a simple web application.

The project follows a complete machine learning pipeline, starting from data handling to model deployment.

---

## Dataset and Initial Processing

The dataset used consists of grayscale images of Braille characters. Unlike standard datasets, this dataset does not have class-wise folders. Instead, the label of each image is embedded in the file name (for example, `a1.jpg`, `b3rot.jpg`, etc.).

The first step was to:

* Read all image files from the directory
* Extract labels from the first character of each filename
* Convert images into grayscale arrays
* Resize all images to a fixed size of 28x28

This ensured uniformity across the dataset before feeding it into the model.

---

## Data Preprocessing

After loading the images:

* Pixel values were normalized from range [0, 255] to [0, 1]
* Data was reshaped to include a channel dimension required for CNN input
* Labels were encoded into numerical format using LabelEncoder

The dataset was then split into:

* Training set (70%)
* Validation set (15%)
* Test set (15%)

This separation was important to properly evaluate model performance and avoid data leakage.

---

## Model Development

A Convolutional Neural Network (CNN) was used for this task because it is well-suited for image-based pattern recognition.

The architecture included:

* Convolutional layers to extract spatial features
* MaxPooling layers to reduce dimensionality
* Fully connected layers for classification
* Dropout layers to reduce overfitting
* Softmax activation in the output layer for multi-class classification (26 classes)

---

## Training and Optimization

The model was initially trained for a fixed number of epochs. During training, it was observed that the model started overfitting after a few epochs.

To handle this:

* EarlyStopping was introduced to stop training when validation loss stopped improving
* Dropout was adjusted to balance bias and variance
* Learning rate was tuned to improve convergence

An important observation was that additional data augmentation reduced performance because the dataset already contained augmented samples and Braille patterns are sensitive to distortion.

---

## Evaluation

The model was evaluated using:

* Test accuracy
* Confusion matrix
* Classification report (precision, recall, F1-score)

The final model achieved:

* Training accuracy around 94%
* Validation accuracy around 88–92%

Misclassified samples were also analyzed to understand model limitations and improve performance.

---

## Prediction Pipeline

A prediction pipeline was created to:

* Take an input image
* Apply the same preprocessing steps used during training
* Pass the image through the trained model
* Convert the predicted class index back to the corresponding alphabet

This pipeline was tested both on dataset images and manually uploaded images.

---

## Deployment

The trained model was saved as a `.h5` file and deployed using Streamlit.

The application allows users to:
* Upload a Braille image
* View the uploaded image
* Get the predicted character along with confidence score

This step ensured that the model could be used in a real-world interactive setting.

---


## Key Takeaways

* Worked with an unstructured dataset where labels had to be extracted manually
* Built a complete CNN pipeline from preprocessing to deployment
* Understood and handled overfitting and underfitting
* Gained experience in deploying machine learning models as web applications

---

## Future Work

* Extend model to support multiple Braille characters in a single image
* Convert sequences of characters into words
* Add text-to-speech output
* Improve UI and user interaction

---
