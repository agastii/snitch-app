import streamlit as st
import pickle
import pandas as pd
import re

# Load models and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('logistic_model.pkl', 'rb') as f:
    LR = pickle.load(f)
with open('tree_model.pkl', 'rb') as f:
    DT = pickle.load(f)
with open('forest_model.pkl', 'rb') as f:
    RF = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Apply custom CSS to style the button
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #C0C0C0 !important;
        color: black !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: white !important;
        color: black !important;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# App content
st.title("📰 Fake News Detector")
st.write("Enter a news article and click **Detect** to check if it's real or fake.")

text_input = st.text_area("News Text", height=150)

if st.button("Detect"):
    if not text_input.strip():
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(text_input)
        vect = vectorizer.transform([cleaned])

        pred_LR = LR.predict(vect)[0]
        pred_DT = DT.predict(vect)[0]
        pred_RF = RF.predict(vect)[0]

        # Majority vote
        result = 1 if [pred_LR, pred_DT, pred_RF].count(1) > 1 else 0

        if result == 1:
            st.success("✅ This appears to be **Real News**.")
        else:
            st.error("🚨 This appears to be **Fake News**.")
