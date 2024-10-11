import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# To load the pre-trained model
model = load_model("mod1.h5")  

def preprocess_image(image):
    image = image.resize((128, 128))

    image_array = np.array(image)

    # Normalize the pixel values
    image_array = image_array / 255.0

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def main():
    st.title("Brain MRI Image Classification")

    uploaded_file = st.file_uploader("Upload a brain MRI image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        preprocessed_image = preprocess_image(image)

        prediction = model.predict(preprocessed_image)

        predicted_class = np.argmax(prediction)

        st.image(image, caption="Uploaded Image")
        if predicted_class == 0:
            st.write("Prediction: Normal")
        else:
            st.write("Prediction: Abnormal")

if __name__ == "__main__":
    main()