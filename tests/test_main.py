"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import pytest
from litestar.testing import TestClient

from app.model_utils import predict_churn
from main import app
# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


def test_predict_churn():
    features = [600, 40, 2, 0.0, 2, 1, 1, 50000.0, "France", "Male"]
    result = predict_churn(features)
    assert result in [0, 1]


def test_predict_churn_invalid_input():
    features = [600, 40, 2, 0.0, 2, 1, 1, 50000.0, "France"]  # Missing one feature

    with pytest.raises(ValueError):
        predict_churn(features)


def test_post_predict_invalid_input():
    with TestClient(app=app) as client:
        # missing the 'Tenure' field completely
        bad_payload = {
            "CreditScore": 600,
            "Age": 40,
            "Balance": 0.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0,
            "Geography": "France",
            "Gender": "Male",
        }
        response = client.post("/predict", json=bad_payload)

        # litestar validation errors return 400
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


def test_post_predict():
    with TestClient(app=app) as client:
        payload = {
            "CreditScore": 600,
            "Age": 40,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0,
            "Geography": "France",
            "Gender": "Male",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 201

        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]


def test_get_health():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_get_home():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Churn Prediction API!"}
