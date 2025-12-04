# app.py
import os
import logging
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# ---------- Logging (setup early) ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cervi_backend")

# ---------- Configuration ----------
# Try multiple possible paths for the model file
def find_model_path():
    """Find the model file in various possible locations."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    possible_paths = [
        # Relative to app.py location (most common)
        os.path.join(app_dir, "backend", "xgb_cervical_pipeline.pkl"),
        # Current working directory
        os.path.join(cwd, "backend", "xgb_cervical_pipeline.pkl"),
        # Simple relative path
        "backend/xgb_cervical_pipeline.pkl",
        # If app.py is in a subdirectory
        os.path.join(os.path.dirname(app_dir), "backend", "xgb_cervical_pipeline.pkl"),
        # Absolute path from cwd
        os.path.join(cwd, "cerviBOT", "backend", "xgb_cervical_pipeline.pkl"),
        # Also try directly in app_dir
        os.path.join(app_dir, "xgb_cervical_pipeline.pkl"),
    ]
    
    logger.info(f"Searching for model file. App dir: {app_dir}, CWD: {cwd}")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            logger.info(f"Found model at: {abs_path}")
            return abs_path
        else:
            logger.debug(f"Checked (not found): {abs_path}")
    
    # Log warning with all checked paths
    checked_paths = [os.path.abspath(p) for p in possible_paths]
    logger.warning(f"Model not found. Checked paths: {checked_paths}")
    return None  # Return None instead of a non-existent path

MODEL_PATH = find_model_path()

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


# ---------- Model holder ----------
model = None
model_path = None


def try_load_model(path: str):
    """Attempt to load a joblib model from path. Returns (model_obj, path) or (None, None) on failure."""
    try:
        logger.info(f"Attempting to load model from: {path}")
        m = joblib.load(path)
        if not (hasattr(m, "predict") or hasattr(m, "predict_proba")):
            logger.warning("Loaded object does not have predict/predict_proba. Not using it.")
            return None, None
        logger.info("Model loaded successfully.")
        return m, path
    except Exception as e:
        logger.exception(f"Failed to load model at {path}: {e}")
        return None, None


# Try load at module import time
if MODEL_PATH and os.path.exists(MODEL_PATH):
    model, model_path = try_load_model(MODEL_PATH)
    if model is None:
        logger.warning("Model file exists but failed to load. Use /upload-model to upload a valid model.")
else:
    logger.info(f"Model file not found. Use /upload-model to upload one or place it at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'xgb_cervical_pipeline.pkl')}")


# ---------- App & CORS ----------
app = FastAPI(title="Cervical Cancer Risk Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Startup event ----------
@app.on_event("startup")
async def startup_event():
    """Try to load the model on startup (fallback if not loaded at import time)."""
    global model, model_path
    if model is None:
        logger.info("Model not loaded at import time. Attempting to load on startup...")
        # Try to find and load the model again
        found_path = find_model_path()
        if found_path and os.path.exists(found_path):
            logger.info(f"Startup: Attempting to load model from {found_path}")
            loaded_model, loaded_path = try_load_model(found_path)
            if loaded_model is not None:
                model = loaded_model
                model_path = loaded_path
                logger.info("Startup: Model loaded successfully!")
            else:
                logger.warning("Startup: Model file found but failed to load.")
        else:
            logger.warning("Startup: Model file not found. Use /upload-model to upload one.")
    else:
        logger.info(f"Model already loaded from: {model_path}")


# ---------- Pydantic input schema ----------
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


# ---------- Helpers ----------
def risk_bucket(proba: float) -> str:
    if proba < 0.33:
        return "Low"
    elif proba < 0.67:
        return "Medium"
    else:
        return "High"


def map_user_to_df(user: UserOptions) -> pd.DataFrame:
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
    df = pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])
    return df


# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Frontend file not found</h1>", status_code=404)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path or "",
    }


@app.post("/predict")
def predict(options: UserOptions) -> Dict[str, Any]:
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Use /upload-model or place model at configured path.")

    try:
        X = map_user_to_df(options)
    except Exception as e:
        logger.exception("Invalid input mapping")
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
            prob_source = "predict_proba"
        else:
            pred = model.predict(X)[0]
            proba = float(pred)
            prob_source = "predict (fallback)"
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    bucket = risk_bucket(proba)
    if bucket == "Low":
        advice = "Low risk — routine screening as per local guidelines is recommended."
    elif bucket == "Medium":
        advice = "Medium risk — consider scheduling a clinical check-up and follow-up screening."
    else:
        advice = "High risk — seek urgent clinical evaluation and further diagnostic testing."

    feature_imp = None
    try:
        if hasattr(model, "feature_importances_"):
            feature_imp = getattr(model, "feature_importances_").tolist()
        elif hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_importances_"):
                    feature_imp = step.feature_importances_.tolist()
                    break
    except Exception:
        feature_imp = None

    return {
        "probability": proba,
        "probability_source": prob_source,
        "risk_bucket": bucket,
        "advice": advice,
        "feature_importances_estimator": feature_imp,
    }


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a joblib model (.pkl/.joblib). Saves next to this app and loads it."""
    global model, model_path
    try:
        contents = await file.read()
        safe_name = os.path.basename(file.filename or "")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        ext = os.path.splitext(safe_name)[1].lower()
        allowed_ext = {".pkl", ".joblib", ".model", ".sav"}
        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), safe_name)
        with open(target_path, "wb") as f:
            f.write(contents)

        loaded, loaded_path = try_load_model(target_path)
        if loaded is None:
            try:
                os.remove(target_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid model or failed to load.")
        model = loaded
        model_path = loaded_path
        logger.info(f"Model uploaded and loaded from {loaded_path}")
        return {"message": "Model uploaded successfully", "model_path": model_path}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")


# ---------- Run server ----------
if __name__ == "__main__":
    # Use environment variables for production, defaults for development
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload)