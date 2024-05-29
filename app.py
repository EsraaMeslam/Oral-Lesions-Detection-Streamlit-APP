# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from PIL import Image

# # Load your model
# model = tf.keras.models.load_model('CNN_mdl.h5')

# def preprocess_image(image):
#     # Resize the image to 304x304
#     image = image.resize((304, 304))
#     img_array = np.array(image)
#     # Normalize the image array to match the model's expected input range
#     # img_array = img_array / 255.0
#     return img_array

# def predict_oral_detection(image):
#     processed_image = preprocess_image(image)
#     # Add batch dimension
#     processed_image = np.expand_dims(processed_image, axis=0)
#     y_pred = model.predict(processed_image)
#     # Convert predicted probabilities to class labels
#     y_pred_labels = np.argmax(y_pred, axis=1)
#     # Get confidence
#     confidence = np.max(y_pred)
#     return y_pred_labels, confidence

# def main():
#     st.write("""
#     <div style='text-align: center;'>
#         <h1>Oral Lesions Detection APP</h1>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.image("oral_img.jpg", use_column_width=True)

#     st.write("""
             

#     This is a simple and easy-to-use web application that utilizes the power of deep learning to detect whether an oral detection is benign or malignant from an uploaded image. 

#     Simply upload an image of an oral detection and click the 'Predict' button to get the prediction result. My model is trained to provide accurate results, helping you to make informed decisions about your oral health.
    
#     Let's predict together and take a step towards a healthier tomorrow! 💪👩‍💻
#     """)

#     uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         if st.button('Predict'):
#             y_pred_labels, confidence = predict_oral_detection(image)
#             # Map class labels to corresponding class names
#             class_names = ['Benign', 'Malignant']
#             y_pred_class_names = [class_names[label] for label in y_pred_labels]
#             # Display prediction result with confidence
#             if y_pred_labels[0] == 0:
#                 st.write(f"The oral detection is predicted as: <u> <b><span style='color:green'>{y_pred_class_names[0]}</span></b></u> with Confidence: {round(confidence * 100, 2)}%", unsafe_allow_html=True)
#                 st.write("<span style='color:green'>Don't worry, everything looks good!❤️</span>", unsafe_allow_html=True)
#             else:
#                 st.write(f"The oral detection is predicted as:<u> <b><span style='color:red'>{y_pred_class_names[0]}</span></b></u> with Confidence: {round(confidence * 100, 2)}%", unsafe_allow_html=True)
#                 st.write("<span style='color:red'>Please consult your doctor for further evaluation.❗️❗️</span>", unsafe_allow_html=True)
#                 st.write("Remember that early detection saves lives. You're taking the right step by seeking medical advice. Stay positive and take care of yourself! ❤️")

# if __name__ == '__main__':
#     main()


import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import InputLayer

# Load your model with custom objects
model = tf.keras.models.load_model('CNN_model__.h5', custom_objects={'InputLayer': InputLayer})

def preprocess_image(image):
    # Resize the image to 304x304
    image = image.resize((304, 304))
    img_array = np.array(image)
    # Normalize the image array to match the model's expected input range
    # img_array = img_array / 255.0
    return img_array

def predict_oral_detection(image):
    processed_image = preprocess_image(image)
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    y_pred = model.predict(processed_image)
    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    # Get confidence
    confidence = np.max(y_pred)
    return y_pred_labels, confidence

def main():
    st.write("""
    <div style='text-align: center;'>
        <h1>Oral Lesions Detection APP</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("oral_img.jpg", use_column_width=True)

    st.write("""
          

    This is a simple and easy-to-use web application that utilizes the power of deep learning to detect whether an oral detection is benign or malignant from an uploaded image. 

    Simply upload an image of an oral detection and click the 'Predict' button to get the prediction result. My model is trained to provide accurate results, helping you to make informed decisions about your oral health.
    
    Let's predict together and take a step towards a healthier tomorrow! 💪👩‍💻
    """)

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            y_pred_labels, confidence = predict_oral_detection(image)
            # Map class labels to corresponding class names
            class_names = ['Benign', 'Malignant']
            y_pred_class_names = [class_names[label] for label in y_pred_labels]
            # Display prediction result with confidence
            if y_pred_labels[0] == 0:
                st.write(f"The oral detection is predicted as: <u> <b><span style='color:green'>{y_pred_class_names[0]}</span></b></u> with Confidence: {round(confidence * 100, 2)}%", unsafe_allow_html=True)
                st.write("<span style='color:green'>Don't worry, everything looks good!❤️</span>", unsafe_allow_html=True)
            else:
                st.write(f"The oral detection is predicted as:<u> <b><span style='color:red'>{y_pred_class_names[0]}</span></b></u> with Confidence: {round(confidence * 100, 2)}%", unsafe_allow_html=True)
                st.write("<span style='color:red'>Please consult your doctor for further evaluation.❗️❗️</span>", unsafe_allow_html=True)
                st.write("Remember that early detection saves lives. You're taking the right step by seeking medical advice. Stay positive and take care of yourself! ❤️")

if __name__ == '__main__':
    main()
