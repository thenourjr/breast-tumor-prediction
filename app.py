import streamlit as st
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import ast

# Load encoders
encoders = pd.read_csv("C:/Users/nourj/Downloads/project/encoders.csv")
encoders_dict = {}
for column in encoders.columns:
    encoders_dict[column] = dict(encoders[column].dropna().to_dict())

image_view = []
left_or_right_breast = []
calc_type = []
calc_distribution = []
for feature, data in encoders_dict.items():
    if 1 in data:
        feature_dict_str = data[1].replace('nan', 'None')
        feature_dict = eval(feature_dict_str)
        feature_list = list(feature_dict.keys())
        globals()[feature] = feature_list

# Load models
binary_model = load_model("C:/Users/nourj/Downloads/project/binary_classification_model.h5")
multi_label_model = load_model("C:/Users/nourj/Downloads/project/multi_label_classification_model.h5")

# Helper functions
def preprocess_image(uploaded_image):
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels (RGB)
    img = cv2.resize(img, (224, 224))  # Resize to the correct shape
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img

def predict_malignant_or_benign(image):
    binary_prediction = binary_model.predict(image)
    return "Malignant" if binary_prediction <= 0.45 else "Benign"

def predict_other_features(image):
    multi_label_prediction = multi_label_model.predict(image)

    predictions = []
    for idx, prediction in enumerate(multi_label_prediction):
        if idx == 5:
            predictions.append("assessment number: " + str(np.argmax(prediction)))
        elif idx == 4:
            predictions.append("breast_density number: " + str(np.argmax(prediction)))
        else:
            max_index = np.argmax(prediction)
            if idx == 0:
                predictions.append(f"image_view: {image_view[max_index]}")
            elif idx == 1:
                predictions.append(f"left_or_right_breast: {left_or_right_breast[max_index]}")
            elif idx == 2:
                predictions.append(f"calc_type: {calc_type[max_index]}")
            elif idx == 3:
                predictions.append(f"calc_distribution: {calc_distribution[max_index]}" if calc_distribution else "calc_distribution: None")

    return predictions

# Streamlit app
st.title("Breast Cancer Detection")
st.write("Upload a breast tumor image for prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    st.subheader("Malignant or Benign Prediction:")
    result = predict_malignant_or_benign(image)
    st.write(f"Prediction: {result}")

    if result == "Malignant":
        st.subheader("Other Features Prediction (Multi-label):")
        features = predict_other_features(image)

        for feature in features:
            st.write(f"- {feature}")

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

st.markdown("""
    <style>
        .css-1v0mbdj {background-color: #f7f7f7; padding: 10px; border-radius: 10px;}
        h1 {color: #6a1b9a;}
        h2 {color: #d32f2f;}
        .stButton button {background-color: #6a1b9a; color: white; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)
