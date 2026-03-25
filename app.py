import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("disease_dl_model.h5")

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Load label encoder
encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("AI Medical Diagnosis Assistant")

symptoms = st.text_input("Enter your symptoms")

if st.button("Predict Disease"):

    seq = tokenizer.texts_to_sequences([symptoms])
    padded = pad_sequences(seq, maxlen=50)

    prediction = model.predict(padded)

    pred_class = np.argmax(prediction)

    # Convert class number → disease name
    disease = encoder.inverse_transform([pred_class])[0]

    confidence = np.max(prediction)

    st.success(f"Predicted Disease: {disease}")
    st.write(f"Confidence: {confidence:.2f}")