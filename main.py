import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_baldness(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0    # Convert to array and scale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        return "Not Bald"
    else:
        return "Bald"

def main():
    st.title("Baldness Detection")

    # Load the model
    model_path = 'baldness_detection_model.h5'  # Adjust the path accordingly
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image_path = "bald-image1.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        st.write("")

        prediction = predict_baldness(image_path, model)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
