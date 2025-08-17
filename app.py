import streamlit as st
import pandas as pd
import joblib
import os

# ====== Load Model and Scaler ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

try:
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# ====== Streamlit Page Config ======
st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")
st.title("🧩 Customer Segmentation App")

# ====== Tabs ======
tab1, tab2 = st.tabs(["📥 Data Input", "📊 Prediction Result"])

# ====== Input Tab ======
with tab1:
    st.subheader("Enter Customer RFM and Country Info")

    col1, col2 = st.columns(2)

    with col1:
        Recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        Frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)

    with col2:
        Monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)
        Country_Encoded = st.selectbox(
            "Country", 
            ['Australia','Austria','Belgium','Canada','Channel Islands','Cyprus',
             'France','Germany','Ireland','Italy','Netherlands','Spain','Sweden',
             'Switzerland','UK','Unspecified']
        )

    if st.button("✅ Predict Cluster"):
        # ====== Prepare input dictionary ======
        input_dict = {
            "Recency": [float(Recency)],
            "Frequency": [float(Frequency)],
            "Monetary": [float(Monetary)]
        }

        # Add all country columns based on scaler.feature_names_in_
        # This ensures the input has EXACTLY the same columns as during training
        for col in scaler.feature_names_in_[3:]:  # skip Recency, Frequency, Monetary
            input_dict[col] = [1 if col == f"Country_{Country_Encoded}" else 0]

        input_df = pd.DataFrame(input_dict)

        # Ensure column order matches scaler
        input_df = input_df[scaler.feature_names_in_]

        # ====== Scale and Predict ======
        X_scaled = scaler.transform(input_df)
        cluster = kmeans.predict(X_scaled)[0]

        st.session_state["cluster"] = cluster
        st.success("Prediction complete! Check the 'Prediction Result' tab.")

# ====== Result Tab ======
with tab2:
    st.subheader("Cluster Result")
    if "cluster" in st.session_state:
        st.info(f"Customer belongs to Cluster: {st.session_state['cluster']}")
    else:
        st.info("Please enter data in the 'Data Input' tab and click Predict.")
