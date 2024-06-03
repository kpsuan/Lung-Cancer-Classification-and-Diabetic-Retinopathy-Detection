import streamlit as st

# Function to run Diabetic Retinopathy detection
def diabetic_retinopathy_detection():
    import cv2
    import numpy as np
    import tensorflow as tf

    # Load the model once at the start to avoid reloading it every time
    new_model = tf.keras.models.load_model("64x3-CNN.h5")

    def predict_class(img):
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGBImg = cv2.resize(RGBImg, (224, 224))
        st.image(RGBImg, caption='Input Image', use_column_width=True)
        image = np.array(RGBImg) / 255.0
        predict = new_model.predict(np.array([image]))
        per = np.argmax(predict, axis=1)
        if per == 1:
            return 'Diabetic Retinopathy NOT DETECTED'
        else:
            return 'Diabetic Retinopathy DETECTED'

    uploaded_file = st.file_uploader("Upload an image for Diabetic Retinopathy detection", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result = predict_class(img)
        
        # Change text color based on prediction result
        if "NOT DETECTED" in result:
            color = "green"
        else:
            color = "red"
        
        st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)

# Function to run Lung Cancer detection
def lung_cancer_detection():
    import numpy as np
    from keras.models import load_model
    from keras.preprocessing import image

    # Load the trained model
    model = load_model("model_vgg16.h5")

    # Define a function to preprocess an image for prediction
    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))  
        img = image.img_to_array(img)  
        img = np.expand_dims(img, axis=0)  
        img = img / 255.0  
        return img

    # Define a dictionary mapping class indices to class labels
    class_labels = {0: "Benign", 1: "Malignant", 2: "Normal"}

    uploaded_file = st.file_uploader("Upload an image for Lung Cancer detection", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        input_image = preprocess_image(uploaded_file)
        predictions = model.predict(input_image)
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        
        # Change text color based on prediction result
        if predicted_label == "Normal":
            color = "green"
        elif predicted_label == "Benign":
            color = "orange"
        else:
            color = "red"

        st.markdown(f"<h2 style='color: {color};'>Predicted Label: {predicted_label}</h2>", unsafe_allow_html=True)

# Streamlit app
st.markdown("<br>", unsafe_allow_html=True)  # Add space before the title
st.title("Medical Image Detection")

st.markdown("**<span style='font-size: 24px;'>Final Project in CMSC 174: Computer Vision</span>**", unsafe_allow_html=True)
st.markdown("<span style='font-size: 14px;'>Name: Sheryl Betonio | Kismet Suan</span>", unsafe_allow_html=True)

# Project description
st.markdown("""
### <span style="font-size: 24px;">Project Description</span>
This application allows users to upload medical images for the detection of Diabetic Retinopathy or Lung Cancer. 
The app can analyze the uploaded images and provide predictions on 
whether the disease is detected or not. Select the disease you want to detect from the dropdown menu and upload an image to get started.
""", unsafe_allow_html=True)

# Increase font size for the select box label
st.markdown("""
### <span style="font-size: 24px;">Which disease would you like to detect?</span>
""", unsafe_allow_html=True)

option = st.selectbox(
    '',
    ('Diabetic Retinopathy', 'Lung Cancer')
)

if option == 'Diabetic Retinopathy':
    diabetic_retinopathy_detection()
elif option == 'Lung Cancer':
    lung_cancer_detection()
