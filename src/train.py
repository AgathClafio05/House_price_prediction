
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load training data
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze("columns")


# Features used by the model
numeric_features = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk"
]

categorical_features = [
    "location",
    "area_type"
]

all_features = numeric_features + categorical_features

# Select only the required features.
# Availability will not be used.
X_train = X_train[all_features].copy()


# Preprocessing for numeric features
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# Preprocessing for categorical features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Create the complete model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# Train the model
model.fit(X_train, y_train)


# Save the trained model
model_path = MODEL_DIR / "house_price_model.joblib"
joblib.dump(model, model_path)

print("Model training completed successfully!")
print(f"Model saved at: {model_path}")
print(f"Training rows: {len(X_train)}")
print(f"Features used: {all_features}")
