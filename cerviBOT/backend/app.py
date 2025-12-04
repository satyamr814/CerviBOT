import uvicorn
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import logging
from fastapi.middleware.cors import CORSMiddleware
import os

# configure logging early so load errors are captured
logging.basicConfig(level=logging.INFO)

# ----- Configuration -----
MODEL_CANDIDATES = [
    "xgb_cervical_pipeline.pkl",
    "model.pkl",
    "xgb_model.pkl",
]

MODEL_DIR = os.path.dirname(__file__)  # project folder
model = None
model_path = None

# Expected feature order for the model's preprocessing pipeline.
# This list must match the column names expected by the saved pipeline.
FEATURE_ORDER = [
    'Age',
    'Num of sexual partners',
    '1st sexual intercourse (age)',
    'Num of pregnancies',
    'Smokes (years)',
    'Hormonal contraceptives',
    'Hormonal contraceptives (years)',
    'STDs:HIV',
    'Pain during intercourse',
    'Vaginal discharge (type- watery, bloody or thick)',
    'Vaginal discharge(color-pink, pale or bloody)',
    'Vaginal bleeding(time-b/w periods , After sex or after menopause)',
]

def load_model_from_candidates():
    global model, model_path
    for name in MODEL_CANDIDATES:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            try:
                logging.info(f"Loading model from {path}")
                model = joblib.load(path)
                model_path = path
                logging.info("Model loaded successfully.")
                return
            except Exception as e:
                logging.exception(f"Failed to load model at {path}: {e}")
    model = None
    model_path = None
    logging.warning("No model file found among candidates.")

# call load after logging setup
load_model_from_candidates()

app = FastAPI(title="Cervical Cancer Risk Chatbot Backend")

# enable CORS for local development / frontend calls (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic Models -----
class UserOptions(BaseModel):
    Age: int
    Num_of_sexual_partners: int
    First_sex_age: int
    Num_of_pregnancies: int
    Smokes_years: float
    Hormonal_contraceptives: str
    Hormonal_contraceptives_years: float
    STDs_HIV: str
    Pain_during_intercourse: str
    Vaginal_discharge_type: str
    Vaginal_discharge_color: str
    Vaginal_bleeding_timing: str

# ----- Helper: categorize probability into risk bucket -----
def risk_bucket(proba: float) -> str:
    if proba < 0.33:
        return "Low"
    elif proba < 0.67:
        return "Medium"
    else:
        return "High"

# ----- Helper: map incoming JSON to DataFrame matching pipeline features -----
def map_user_to_df(user: UserOptions) -> pd.DataFrame:
    # Convert keys to the exact column names expected in the pipeline
    row = {
        'Age': int(user.Age),
        'Num of sexual partners': int(user.Num_of_sexual_partners),
        '1st sexual intercourse (age)': int(user.First_sex_age),
        'Num of pregnancies': int(user.Num_of_pregnancies),
        'Smokes (years)': float(user.Smokes_years),
        'Hormonal contraceptives': str(user.Hormonal_contraceptives),
        'Hormonal contraceptives (years)': float(user.Hormonal_contraceptives_years),
        'STDs:HIV': str(user.STDs_HIV),
        'Pain during intercourse': str(user.Pain_during_intercourse),
        'Vaginal discharge (type- watery, bloody or thick)': str(user.Vaginal_discharge_type),
        'Vaginal discharge(color-pink, pale or bloody)': str(user.Vaginal_discharge_color),
        'Vaginal bleeding(time-b/w periods , After sex or after menopause)': str(user.Vaginal_bleeding_timing)
    }
    # keep only features pipeline expects, and preserve order
    df = pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])
    return df

# ----- Predict endpoint -----
@app.post("/predict")
def predict(options: UserOptions) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Upload a model or place one of the expected files in the project folder.")

    # Map and validate types / logical constraints
    try:
        X = map_user_to_df(options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Run predict_proba with fallback to predict
    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])   # positive class probability
            prob_source = "predict_proba"
        else:
            # fallback: use predicted label as a coarse probability (0.0 or 1.0)
            pred = model.predict(X)[0]
            proba = float(pred)
            prob_source = "predict (fallback)"
    except Exception as e:
        logging.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    bucket = risk_bucket(proba)
    # build friendly message & action
    if bucket == "Low":
        advice = "Low risk — routine screening as per local guidelines is recommended."
    elif bucket == "Medium":
        advice = "Medium risk — consider scheduling a clinical check-up and follow-up screening."
    else:
        advice = "High risk — seek urgent clinical evaluation and further diagnostic testing."

    # Optional: feature importance or partial explanation (lightweight)
    feature_imp = None
    try:
        if hasattr(model, "named_steps") and 'model' in model.named_steps:
            est = model.named_steps['model']
            if hasattr(est, "feature_importances_"):
                feature_imp = est.feature_importances_.tolist()
    except Exception:
        feature_imp = None

    response = {
        "probability": proba,
        "probability_source": prob_source,
        "risk_bucket": bucket,
        "advice": advice,
        "feature_importances_estimator": feature_imp
    }
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": model_path}

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)