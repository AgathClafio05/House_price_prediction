# Bengaluru House Price Prediction

An end-to-end Machine Learning project that predicts residential property prices in Bengaluru using regression techniques. The project covers data analysis, data preprocessing, model training, evaluation, and deployment through a Streamlit web application.

## Project Overview

The objective of this project is to estimate house prices based on property characteristics such as location, total square footage, number of bedrooms, bathrooms, balconies, area type, and availability.

The trained regression model is integrated into a Streamlit application, allowing users to enter property details and receive an estimated house price.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Missing-value handling and duplicate removal
- Square-footage conversion and feature engineering
- Outlier investigation while retaining price outliers
- Categorical feature encoding using One-Hot Encoding
- Regression model training using Linear Regression
- Model evaluation using MAE, RMSE, and R²
- Interactive house-price prediction using Streamlit
- Model serialization and reuse for predictions

## Tech Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** VS Code
- **Version Control:** Git and GitHub

## Dataset

The project uses a Bengaluru house-price dataset containing property information.

Important features include:

| Feature | Description |
|---|---|
| `area_type` | Type of property area |
| `availability` | Property availability status |
| `location` | Property location in Bengaluru |
| `total_sqft` | Total property area in square feet |
| `bhk` | Number of bedrooms |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | House price in lakhs; prediction target |

The original dataset also contains a `society` column, which is excluded during preprocessing in this version.

## Project Workflow

1. **Project Setup:** Configure the development environment and dependencies.
2. **Data Understanding:** Inspect dataset dimensions, columns, data types, missing values, and duplicates.
3. **Exploratory Data Analysis:** Analyze distributions, price patterns, feature relationships, and potential outliers.
4. **Data Cleaning:** Handle missing values, duplicates, inconsistent formats, and invalid records.
5. **Feature Preparation:** Separate features and target, then create training and testing datasets.
6. **Model Training:** Train a Linear Regression model using a preprocessing pipeline.
7. **Model Evaluation:** Evaluate predictions using MAE, RMSE, and R².
8. **Model Saving and Verification:** Save the trained pipeline and test predictions on new inputs.
9. **Application Development:** Build an interactive Streamlit interface.
10. **Testing and Deployment:** Validate the application and prepare it for hosting.

## Project Structure

```text
bengaluru-house-price-prediction/
│
├── data/
│   ├── bengaluru_house_prices.csv
│   ├── bengaluru_house_prices_cleaned.csv
│   └── splits/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── data_cleaning.ipynb
│   └── feature_preparation.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── test_model.py
│
├── models/
│   └── house_price_model.joblib
│
├── reports/
│   ├── evaluation_metrics.csv
│   ├── prediction_results.csv
│   └── model_verification.csv
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd bengaluru-house-price-prediction
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Evaluate the model

```bash
python src/evaluate.py
```

### 6. Verify predictions

```bash
python src/test_model.py
```

### 7. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Model Evaluation

The model is evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures how well the model explains variation in the test-set house prices relative to a mean baseline.

Actual evaluation results should be recorded after running `src/evaluate.py`.

| Metric | Result |
|---|---|
| MAE | Add your measured result |
| RMSE | Add your measured result |
| R² Score | Add your measured result |

## Deployment

The Streamlit application can be deployed using a compatible hosting platform, such as Streamlit Community Cloud.

Before deployment:

1. Confirm that the application runs locally.
2. Include all required dependencies in `requirements.txt`.
3. Ensure the trained model is available at the expected path.
4. Ensure the required dataset is available if the application loads dropdown options from it.
5. Configure the deployment to run `app.py`.

## Limitations

- Predictions depend on the quality and coverage of the training data.
- Linear Regression may not capture complex relationships between location, area, and property prices.
- Price outliers are retained and may influence model performance.
- Predictions are estimates, not guaranteed selling prices.
- Model performance should be assessed on unseen test data before real-world use.

## Future Improvements

- Compare Linear Regression with Random Forest and other suitable regression models.
- Improve location and square-footage feature engineering.
- Experiment with transformations for skewed price distributions.
- Add cross-validation and hyperparameter tuning.
- Improve input validation and prediction monitoring.
- Deploy the application and add screenshots of the working interface.

## Author

**Agath Clafio**

Machine Learning | Data Analysis | Python

GitHub: <YOUR_GITHUB_PROFILE_URL>

---

*This project was developed for practical learning in end-to-end Machine Learning, regression modeling, and deployment.*
