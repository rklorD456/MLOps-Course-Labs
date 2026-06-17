"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import pandas as pd
import warnings
import joblib

# tell python to ignore the harmless scikit-learn warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# TODO 1: Load your serialized churn model (and preprocessor if any) from data/
try:
    model = joblib.load("data/model.pkl")

except FileNotFoundError:
    raise FileNotFoundError(
        "Model file not found. Please ensure 'model.pkl' exists in the 'data/' directory."
    )

try:
    preprocessor = joblib.load("data/preprocessor.pkl")
except FileNotFoundError:
    preprocessor = None


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    # TODO 3: Preprocess the features

    column_names = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography",
        "Gender",
    ]

    raw_df = pd.DataFrame([features], columns=column_names)
    processed_features = preprocessor.transform(raw_df)

    # TODO 4: Use model.predict() on processed_features to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array

    prediction = model.predict(processed_features)
    return int(prediction[0])


if __name__ == "__main__":
    # TODO 5: Replace with sample features that match your model
    sample = [619, 42, 2, 0.0, 1, 1, 1, 101348.88, "France", "Female"]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
