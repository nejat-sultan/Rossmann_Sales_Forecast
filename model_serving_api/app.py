# app.py
from fastapi import FastAPI # type: ignore
import joblib # type: ignore
import numpy as np
from pydantic import BaseModel # type: ignore

app = FastAPI()

model_path = 'C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/models/random_forest_model.pkl'
model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)  
    prediction = model.predict(features)
    return {"prediction": prediction[0]}

