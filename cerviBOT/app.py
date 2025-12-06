# app.py
import os
import logging
from typing import Dict, Any, Optional
import base64
import io

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import preprocessing module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from preprocess import preprocess_input, validate_input

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
        # New model location (primary) - relative to app.py
        os.path.join(app_dir, "model_files", "cervical_cancer_model.pkl"),
        # Current working directory (Render uses rootDir: cerviBOT)
        os.path.join(cwd, "model_files", "cervical_cancer_model.pkl"),
        # Simple relative path from cwd
        os.path.join(cwd, "cervical_cancer_model.pkl"),
        # If app.py is in cerviBOT subdirectory
        os.path.join(app_dir, "..", "model_files", "cervical_cancer_model.pkl"),
        os.path.join(cwd, "cerviBOT", "model_files", "cervical_cancer_model.pkl"),
        # Old model location (backward compatibility)
        os.path.join(app_dir, "backend", "xgb_cervical_pipeline.pkl"),
        os.path.join(cwd, "backend", "xgb_cervical_pipeline.pkl"),
        # Simple relative paths
        "model_files/cervical_cancer_model.pkl",
        "backend/xgb_cervical_pipeline.pkl",
        # If app.py is in a subdirectory
        os.path.join(os.path.dirname(app_dir), "backend", "xgb_cervical_pipeline.pkl"),
        # Absolute path from cwd
        os.path.join(cwd, "cerviBOT", "backend", "xgb_cervical_pipeline.pkl"),
        # Also try directly in app_dir
        os.path.join(app_dir, "xgb_cervical_pipeline.pkl"),
        # Render deployment paths (rootDir is cerviBOT, so cwd should be cerviBOT)
        os.path.join(cwd, "cerviBOT", "model_files", "cervical_cancer_model.pkl"),
        os.path.join(cwd, "cerviBOT", "backend", "xgb_cervical_pipeline.pkl"),
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
        
        # Check if imbalanced-learn is available (required for imblearn pipelines)
        try:
            import imblearn
            logger.debug("imbalanced-learn is available")
        except ImportError as ie:
            logger.error(f"CRITICAL: imbalanced-learn is not installed! This is required for the model. Error: {ie}")
            logger.error("Please install it: pip install imbalanced-learn")
            return None, None
        
        m = joblib.load(path)
        
        # Log model type for debugging
        logger.info(f"Model loaded, type: {type(m).__name__}")
        
        if not (hasattr(m, "predict") or hasattr(m, "predict_proba")):
            logger.warning("Loaded object does not have predict/predict_proba. Not using it.")
            return None, None
        
        # Verify model has required methods
        if hasattr(m, "predict"):
            logger.debug("Model has predict method")
        if hasattr(m, "predict_proba"):
            logger.debug("Model has predict_proba method")
        
        logger.info("Model loaded successfully.")
        return m, path
    except ImportError as ie:
        logger.error(f"Import error while loading model: {ie}")
        logger.error("This usually means a required dependency is missing.")
        logger.error("Required packages: joblib, pandas, scikit-learn, xgboost, imbalanced-learn")
        return None, None
    except Exception as e:
        logger.exception(f"Failed to load model at {path}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return None, None


# Try load at module import time
if MODEL_PATH and os.path.exists(MODEL_PATH):
    model, model_path = try_load_model(MODEL_PATH)
    if model is None:
        logger.warning("Model file exists but failed to load. Use /upload-model to upload a valid model.")
else:
    logger.info(f"Model file not found. Use /upload-model to upload one or place it at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_files', 'cervical_cancer_model.pkl')}")


# ---------- App & CORS ----------
app = FastAPI(title="Cervical Cancer Risk Chatbot Backend", version="2.0.0")
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
                logger.error("Startup: Model file found but failed to load.")
                logger.error("Check the logs above for detailed error information.")
                logger.error("Common issues:")
                logger.error("  1. Missing dependency: pip install imbalanced-learn")
                logger.error("  2. Model file corrupted or incompatible version")
                logger.error("  3. XGBoost version mismatch")
        else:
            logger.warning("Startup: Model file not found. Use /upload-model to upload one.")
    else:
        logger.info(f"Model already loaded from: {model_path}")


# ---------- Pydantic input schema ----------
class UserOptions(BaseModel):
    Age: int = Field(..., ge=0, le=120, description="Age in years")
    Num_of_sexual_partners: int = Field(..., ge=0, description="Number of sexual partners")
    First_sex_age: int = Field(..., ge=0, le=120, description="Age at first sexual intercourse")
    Num_of_pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Smokes_years: float = Field(0.0, ge=0, description="Years of smoking")
    Hormonal_contraceptives: str = Field("No", description="Using hormonal contraceptives (Yes/No)")
    Hormonal_contraceptives_years: float = Field(0.0, ge=0, description="Years of hormonal contraceptive use")
    STDs_HIV: str = Field("No", description="HIV status (Yes/No/Negative/Positive)")
    Pain_during_intercourse: str = Field("No", description="Pain during intercourse (Yes/No)")
    Vaginal_discharge_type: str = Field("None", description="Vaginal discharge type")
    Vaginal_discharge_color: str = Field("normal", description="Vaginal discharge color")
    Vaginal_bleeding_timing: str = Field("None", description="Vaginal bleeding timing")


# ---------- Helpers ----------
def risk_bucket(proba: float) -> str:
    """Categorize risk based on probability."""
    if proba < 0.33:
        return "Low"
    elif proba < 0.67:
        return "Medium"
    else:
        return "High"


def get_risk_color(risk: str) -> str:
    """Get color code for risk level."""
    colors = {
        "Low": "#10b981",      # Green
        "Medium": "#f59e0b",   # Yellow/Orange
        "High": "#ef4444"      # Red
    }
    return colors.get(risk, "#6b7280")


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


@app.get("/ui/test", response_class=HTMLResponse)
def test_ui():
    """Serve the test UI page."""
    test_ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ui.html")
    if os.path.exists(test_ui_path):
        with open(test_ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Test UI file not found</h1>", status_code=404)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path or "",
        "version": "2.0.0"
    }


@app.post("/predict")
def predict(options: UserOptions) -> Dict[str, Any]:
    """Make a prediction based on user input."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Use /upload-model or place model at configured path.")

    # Validate input
    is_valid, error_msg = validate_input(options.dict())
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Preprocess input using the preprocessing module
        X = preprocess_input(options.dict())
    except Exception as e:
        logger.exception("Invalid input preprocessing")
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        # Ensure X has the correct column order
        X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
        
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_ordered)[0][1])
            prob_source = "predict_proba"
        else:
            X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
            pred = model.predict(X_ordered)[0]
            proba = float(pred)
            prob_source = "predict (fallback)"
    except AttributeError as e:
        if "_name_to_fitted_passthrough" in str(e) or "ColumnTransformer" in str(e):
            logger.error("scikit-learn version mismatch! Model was trained with a different version.")
            logger.error("Please update scikit-learn: pip install 'scikit-learn>=1.3.0'")
            raise HTTPException(
                status_code=500, 
                detail="Model compatibility error. Please ensure scikit-learn>=1.3.0 is installed."
            )
        logger.exception("Prediction failed - AttributeError")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    bucket = risk_bucket(proba)
    risk_color = get_risk_color(bucket)
    
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
        "probability_percent": round(proba * 100, 2),
        "probability_source": prob_source,
        "risk_bucket": bucket,
        "risk_color": risk_color,
        "advice": advice,
        "feature_importances_estimator": feature_imp,
        "label": "Positive" if proba >= 0.5 else "Negative",
        "confidence": "High" if abs(proba - 0.5) > 0.3 else "Medium" if abs(proba - 0.5) > 0.15 else "Low"
    }


@app.post("/explain")
def explain_prediction(options: UserOptions) -> Dict[str, Any]:
    """Generate SHAP explanation for the prediction (optional)."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        import shap
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP not installed. Install with: pip install shap")
    
    try:
        # Preprocess input
        X = preprocess_input(options.dict())
        X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
        
        # Get prediction
        proba = float(model.predict_proba(X_ordered)[0][1])
        
        # Create SHAP explainer
        # For tree models, use TreeExplainer
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            xgb_model = model.named_steps['model']
        else:
            xgb_model = model
        
        explainer = shap.TreeExplainer(xgb_model)
        
        # Get SHAP values (need to transform X through pipeline first)
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            X_transformed = model.named_steps['preprocessor'].transform(X_ordered)
            shap_values = explainer.shap_values(X_transformed)
        else:
            shap_values = explainer.shap_values(X_ordered)
        
        # Get feature names
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = list(X_ordered.columns)
        
        # Create feature importance dict
        feature_importance = {}
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        
        for i, name in enumerate(feature_names):
            if i < len(shap_vals[0]):
                feature_importance[str(name)] = float(shap_vals[0][i])
        
        return {
            "probability": proba,
            "feature_importance": feature_importance,
            "message": "SHAP explanation generated successfully"
        }
    except Exception as e:
        logger.exception("SHAP explanation failed")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.get("/example_profiles")
def example_profiles() -> Dict[str, Any]:
    """Return example profiles for testing."""
    return {
        "low_risk": {
            "Age": 25,
            "Num_of_sexual_partners": 1,
            "First_sex_age": 20,
            "Num_of_pregnancies": 0,
            "Smokes_years": 0.0,
            "Hormonal_contraceptives": "No",
            "Hormonal_contraceptives_years": 0.0,
            "STDs_HIV": "No",
            "Pain_during_intercourse": "No",
            "Vaginal_discharge_type": "None",
            "Vaginal_discharge_color": "normal",
            "Vaginal_bleeding_timing": "None"
        },
        "medium_risk": {
            "Age": 35,
            "Num_of_sexual_partners": 3,
            "First_sex_age": 16,
            "Num_of_pregnancies": 2,
            "Smokes_years": 5.0,
            "Hormonal_contraceptives": "Yes",
            "Hormonal_contraceptives_years": 8.0,
            "STDs_HIV": "No",
            "Pain_during_intercourse": "Yes",
            "Vaginal_discharge_type": "watery",
            "Vaginal_discharge_color": "pink",
            "Vaginal_bleeding_timing": "Between periods"
        },
        "high_risk": {
            "Age": 45,
            "Num_of_sexual_partners": 5,
            "First_sex_age": 14,
            "Num_of_pregnancies": 4,
            "Smokes_years": 15.0,
            "Hormonal_contraceptives": "Yes",
            "Hormonal_contraceptives_years": 20.0,
            "STDs_HIV": "Yes",
            "Pain_during_intercourse": "Yes",
            "Vaginal_discharge_type": "bloody",
            "Vaginal_discharge_color": "bloody",
            "Vaginal_bleeding_timing": "After sex"
        }
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

        # Save to model_files directory
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files")
        os.makedirs(model_dir, exist_ok=True)
        target_path = os.path.join(model_dir, safe_name)

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
