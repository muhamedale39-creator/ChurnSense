from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd


artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifacts
    try:
        artifacts = joblib.load("churn_artifacts.joblib")
        print("Successfully loaded churn_artifacts.joblib")
    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
    yield
    artifacts.clear()

app = FastAPI(lifespan=lifespan, title="ChurnSense Prediction API")

class CustomerData(BaseModel):
    gender: int = Field(..., description="1 for Male, 0 for Female")
    SeniorCitizen: int
    Partner: int = Field(..., description="1 for Yes, 0 for No")
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int = Field(..., description="1 for Yes, 0 for No")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: int
    PaymentMethod: str = Field(..., description="Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check")
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    applied_threshold: float

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    
    input_df = pd.DataFrame([customer.model_dump()])
    
    expected_cols = artifacts["feature_columns"]
    input_df = input_df[expected_cols]

    
    transformer = artifacts["transformer"]
    X_processed = transformer.transform(input_df)
 
    model = artifacts["model"]
    threshold = artifacts["threshold"]
    
    prob = model.predict_proba(X_processed)[0, 1]
    
    prediction = int(prob >= threshold)
    
    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=float(prob),
        applied_threshold=float(threshold)
    )