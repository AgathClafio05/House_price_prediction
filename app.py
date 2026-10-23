
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "bengaluru_house_prices_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"


# Page settings
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Bengaluru House Price Prediction")
st.write("Enter the house details to estimate its price in Bengaluru.")


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
except FileNotFoundError as error:
    st.error(f"Required file not found: {error.filename}")
    st.stop()


# Prepare dropdown options
area_types = sorted(
    df["area_type"].dropna().astype(str).unique().tolist()
)

locations = sorted(
    df["location"].dropna().astype(str).unique().tolist()
)


# House details
st.subheader("Enter House Details")

with st.form("house_details_form"):

    area_type = st.selectbox(
        "Area Type",
        area_types
    )

    location = st.selectbox(
        "Location",
        locations
    )

    total_sqft = st.number_input(
        "Total Square Feet",
        min_value=1.0,
        value=1200.0,
        step=100.0
    )

    bhk = st.number_input(
        "Bedrooms (BHK)",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    balcony = st.number_input(
        "Balconies",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    submitted = st.form_submit_button("Predict House Price")


# Predict price
if submitted:

    input_data = pd.DataFrame([{
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk,
        "location": location,
        "area_type": area_type
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])

        if not pd.notna(predicted_price):
            st.error("The model returned an invalid prediction.")
        else:
            st.success("Estimated House Price")
            st.metric(
                label="Predicted Price",
                value=f"₹ {predicted_price:.2f} Lakhs"
            )

            st.caption(
                "This is a machine-learning estimate, not a guaranteed market price."
            )

    except (ValueError, KeyError) as error:
        st.error(f"Unable to predict the price: {error}")
