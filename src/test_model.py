import joblib
import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

new_house = pd.DataFrame([{
    "area_type": "Super built-up Area",
    "availability": "Ready To Move",
    "location": "Whitefield",
    "total_sqft": 1200,
    "bath": 2,
    "balcony": 1,
    "bhk": 2
}])

prediction = model.predict(new_house)[0]

print(f"Predicted house price: ₹{prediction:.2f} lakhs")

assert len(new_house) == 1
assert np.isfinite(prediction), "Prediction is not a valid number"

print("Prediction test passed.")

test_houses = pd.DataFrame([
    {
        "area_type": "Super built-up Area",
        "availability": "Ready To Move",
        "location": "Whitefield",
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1,
        "bhk": 2
    },
    {
        "area_type": "Built-up Area",
        "availability": "Ready To Move",
        "location": "Electronic City",
        "total_sqft": 1500,
        "bath": 3,
        "balcony": 2,
        "bhk": 3
    }
])

predictions = model.predict(test_houses)

test_houses["predicted_price_lakhs"] = predictions

print(test_houses)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

test_houses.to_csv(
    REPORT_DIR / "model_verification.csv",
    index=False
)

print("Verification results saved.")