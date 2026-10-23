import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "splits"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

model = joblib.load(MODEL_PATH)

print("Test features:", X_test.shape)
print("Test target:", y_test.shape)

y_pred = model.predict(X_test)

print("Predictions generated:", len(y_pred))
print("First 10 predictions:", y_pred[:10])

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("========== MODEL EVALUATION ==========")
print(f"MAE: {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R² Score: {r2:.4f}")
print("======================================")

results = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": y_pred
})

results["Absolute_Error"] = abs(
    results["Actual_Price"] - results["Predicted_Price"]
)

print(results.head(20))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

errors = y_test.to_numpy() - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, errors, alpha=0.5)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Price (Lakhs)")
plt.ylabel("Prediction Error (Lakhs)")
plt.title("Prediction Residuals")
plt.tight_layout()
plt.show()

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(REPORT_DIR / "evaluation_metrics.csv", index=False)

results.to_csv(
    REPORT_DIR / "prediction_results.csv",
    index=False
)

print("Evaluation reports saved in:", REPORT_DIR)