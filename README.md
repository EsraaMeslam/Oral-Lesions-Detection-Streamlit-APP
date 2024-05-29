<h1>Oral Lesions Detection APP</h1>

<h3>Overview</h3>
This repository contains the implementation of a deep learning model (CNN) for Oral Lesions Dtection.The model is trained on Dataset, hosted on Kaggle.
<br>
the dataset is from https://data.mendeley.com/datasets/mhjyrn35p4/2, that Published: 5 February 2021 | Version 2 | DOI: 10.17632/mhjyrn35p4.2 Contributors: Chandrashekar H S, Geetha Kiran A, Murali S, Dinesh M S, Nanditha B R
<br>
Dataset link:(https://www.kaggle.com/datasets/mohamedgobara/oral-lesions-malignancy-detection-dataset)<br>
My Code:(https://www.kaggle.com/code/esraameslamsayed/oral-lesions-detection-cnn)
<h3>Description</h3>

The dataset includes color images of oral lesions captured using mobile cameras and intraoral cameras. These images can be used for identifying potential oral malignancies by image analysis. These images have been collected in consultation with doctors from different hospitals and colleges in Karnataka, India. This dataset contains two folders - original_data and augmented_data. The first folder contains images of 165 benign lesions and 158 malignant lesions. The second folder contains images created by augmenting the original images. The augmentation techniques used are flipping, rotation and resizing. 

<h3>How the app Works:</h3>
<ul>
<li>Upload an Image: Simply upload a clear image of the oral lesion you want to analyze. Our intuitive interface makes the process seamless.</li>

<li>Click Predict: Once the image is uploaded, hit the 'Predict' button, and our advanced deep learning model goes to work.</li>

<li>Receive Informed Analysis: Within moments, receive a detailed analysis of the lesion, including whether it's benign or malignant, along with confidence levels for the prediction.</li>
</ul>

<h3>Model Building</h3>
A powerful Convolutional Neural Network (CNN) model is built using TensorFlow and Keras. The model architecture includes multiple Conv2D layers followed by BatchNormalization and MaxPooling2D layers to extract features from the input images.

<h3>Model Training</h3>
The CNN model is trained using a dataset of oral lesion images. The images are preprocessed, including resizing, to match the input size expected by the model. The model is trained to classify the lesions into benign or malignant categories.

<h3>Model Evaluation<\h3> The trained model is evaluated using validation data to assess its performance in terms of accuracy, precision, recall, and other relevant metrics.
<h3>Model Deployment</h3> Once the model is trained and evaluated, it is saved to a file (CNN_model__.h5) for future use. This file is then loaded in the web application for real-time predictions.

<h3>Prediction Process</h3> When a user uploads an image of an oral lesion and clicks the 'Predict' button, the image is passed through the loaded CNN model for prediction. The model predicts whether the lesion is benign or malignant and returns the result along with a confidence score.

<h3>Result Display<\h3> The prediction result, along with the confidence score, is displayed to the user on the web application interface. If the lesion is predicted as malignant, the user is advised to consult a doctor for further evaluation.
<h3>Requirements</h3>
<ul>
<li>numpy==1.24.3</li>
<li>streamlit==1.34.0</li>
<li>tensorflow==2.14.0</li>
<li>Pillow==9.2.0</li>
</ul>






