import streamlit as st
import joblib
import re
import string

# Page configuration
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('disaster_tweet_classifier.pkl')

model = load_model()

# Text cleaning function
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# App UI
st.title("🚨 Disaster Tweet Classifier")
st.write("Enter a tweet below to predict whether it is reporting a real emergency or not.")

# Text input
user_input = st.text_area("Tweet Text", placeholder="e.g., Wildfire reported near the highway, evacuations underway...")

if st.button("Classify Tweet"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        # Preprocess and predict
        cleaned = clean_tweet(user_input)
        pred = model.predict([cleaned])[0]
        
        # If your model supports probability estimates:
        try:
            proba = model.predict_proba([cleaned])[0][1]
        except AttributeError:
            proba = None

        st.divider()
        if pred == 1:
            st.error("⚠️ **Prediction: Disaster-Related Tweet**")
        else:
            st.success("✅ **Prediction: Non-Disaster Tweet**")

        if proba is not None:
            st.write(f"**Disaster Probability:** {proba:.2%}")
