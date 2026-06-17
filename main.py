"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, get, post
from pydantic import BaseModel

from app.logger_setup import setup_logging
from app.model_utils import predict_churn

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography: str
    Gender: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@get("/")
async def home() -> dict:
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API!"}


@get("/health")
async def health() -> dict:
    logger.info("Health endpoint accessed")
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest) -> dict:
    features = [
        data.CreditScore,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
        data.Geography,
        data.Gender,
    ]

    prediction = predict_churn(features)

    logger.info(f"input features: {features} | prediction: {prediction}")
    return {"prediction": prediction}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[home, health, predict],
)
