import streamlit as st
import pandas as pd
import joblib

# Load models & scaler & feature names
kmeans = joblib.load("models/kmeans_model.pkl")
gmm = joblib.load("models/gmm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Extract country list from feature names
countries = [col.replace("Country_", "") for col in feature_names if col.startswith("Country_")]

# Page config
st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="centered")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🛍️ Customer Segmentation App</h1>
    <p style='text-align: center; color: gray;'>Predict customer clusters using <b>KMeans</b> & <b>GMM</b></p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# User inputs with columns
col1, col2 = st.columns(2)
with col1:
    recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, step=1)
    monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0, step=10.0)
with col2:
    frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0, step=1)
    country = st.selectbox("🌍 Country", options=sorted(countries))

st.write("---")

# Prediction button
if st.button("🚀 Predict Cluster", use_container_width=True):
    # Create DataFrame for user input
    user_df = pd.DataFrame([[recency, frequency, monetary, country]],
                           columns=["Recency", "Frequency", "Monetary", "Country"])
    
    # One-hot encode country
    user_encoded = pd.get_dummies(user_df, columns=["Country"])

    # Reindex to match training feature names
    user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale input
    user_scaled = scaler.transform(user_encoded)

    # Predictions
    km_cluster = kmeans.predict(user_scaled)[0]
    gmm_cluster = gmm.predict(user_scaled)[0]

    # Show results nicely with bigger text
    st.markdown("### 🎯 Prediction Results")

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px;'>
            <h2 style='color:#1F618D;'>🔹 KMeans Cluster: <b>{km_cluster}</b></h2>
            <h2 style='color:#117864;'>🔸 GMM Cluster: <b>{gmm_cluster}</b></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("✅ Prediction completed successfully!")
