import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

#load trained model
model = load_model("/teamspace/studios/this_studio/CNN_Mini_Project/cifar10_cnn.h5")


#CIFAR-10 class names
class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image (32x32 RGB) and the model will predict its class")

 #File uploader
uploaded_file = st.file_uploader("Upload an image",type = ['jpg','png','jpeg']) 

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).resize((32,32))
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    # Preprocess
    img_array = np.array(image)
    img_array = img_array.astype("float32")/255.0
    img_array = np.expand_dims(img_array,axis = 0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    st.subheader(f"✅ Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Plot prediction probabilites
    fig,ax = plt.subplots()
    ax.bar(class_names, predictions[0])
    ax.set_ylabel("Confidence")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names,rotation = 45)
    st.pyplot(fig)           