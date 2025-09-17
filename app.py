import streamlit as st
import pandas as pd
import pickle

# Load model
with open("xgb_age_model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

# -------------------
# APP STYLING
# -------------------
st.set_page_config(page_title="Possum Age Prediction", page_icon="🦉", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #6f42c1;
        color:white;
        font-weight:bold;
        border-radius:10px;
        padding:10px 24px;
    }
    .stButton>button:hover {
        background-color:#563d7c;
        color:white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# HEADER
# -------------------
st.title("🦉 Possum Age Prediction App")
st.markdown("This app predicts the **Age** of a possum based on its biological measurements.")

st.divider()

# -------------------
# INPUT FORM
# -------------------
st.subheader("📥 Enter Possum Details")

col1, col2, col3 = st.columns(3)

user_input = {}

# Spread features into columns
for i, col in enumerate(feature_names):
    with [col1, col2, col3][i % 3]:
        if "sex" in col.lower():
            user_input[col] = st.radio(f"{col}", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        elif "site" in col.lower() or "pop" in col.lower():
            user_input[col] = st.selectbox(f"{col}", [0, 1])
        else:
            user_input[col] = st.slider(f"{col}", 0.0, 100.0, 10.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

st.divider()

# -------------------
# PREDICTION
# -------------------
if st.button("🚀 Predict Age"):
    pred_age = model.predict(input_df)[0]
    st.success(f"🎉 Predicted Possum Age: **{pred_age:.2f} years**")
