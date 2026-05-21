"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

# TODO 1: Load your serialized churn model (and preprocessor if any) from data/
model = ...


def preprocess(features: list[float]) -> list[float]:
    """
    Takes raw features and applies necessary preprocessing (e.g. scaling).
    """
    # TODO 2: Apply any preprocessing steps here (if applicable)
    return features


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    # TODO 3: Preprocess the features
    processed_features = preprocess(features)
    
    # TODO 4: Use model.predict() on processed_features to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array
    pass


if __name__ == "__main__":
    # TODO 5: Replace with sample features that match your model
    sample = []
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
