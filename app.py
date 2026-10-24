import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
DATA_PATH = BASE_DIR / "Data" / "bengaluru_house_prices_cleaned.csv"

# Page configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("Bengaluru House Price Prediction")
st.write("Enter the property details to estimate its price.")

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load the model or dataset: {e}")
    st.stop()

# Input fields
area_type = st.selectbox(
    "Area Type",
    sorted(df["area_type"].dropna().unique())
)

availability = st.selectbox(
    "Availability",
    sorted(df["availability"].dropna().unique())
)

location = st.selectbox(
    "Location",
    sorted(df["location"].dropna().unique())
)

total_sqft = st.number_input(
    "Total Square Feet",
    min_value=1.0,
    value=1200.0,
    step=50.0
)

bhk = st.number_input(
    "Number of Bedrooms (BHK)",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

bath = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

balcony = st.number_input(
    "Number of Balconies",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

# Prediction
if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame([{
        "area_type": area_type,
        "availability": availability,
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk
    }])

    try:
        prediction = model.predict(input_data)[0]

        if prediction <= 0:
            st.warning(
                f"The model returned ₹{prediction:.2f} lakhs. "
                "This is not a valid positive house-price estimate."
            )
        else:
            st.success("Prediction completed.")
            st.metric(
                "Estimated House Price",
                f"₹{prediction:.2f} lakhs"
            )
            st.caption(
                "This is a model estimate, not a guaranteed market price."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
