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
    
    # Build comprehensive list of paths to check
    possible_paths = []
    
    # Primary: relative to app.py location
    possible_paths.append(os.path.join(app_dir, "model_files", "cervical_cancer_model.pkl"))
    
    # Current working directory (Render with rootDir: cerviBOT means cwd is the cerviBOT folder)
    possible_paths.append(os.path.join(cwd, "model_files", "cervical_cancer_model.pkl"))
    
    # Simple relative paths (from cwd)
    possible_paths.append("model_files/cervical_cancer_model.pkl")
    possible_paths.append("./model_files/cervical_cancer_model.pkl")
    
    # If app.py is in a subdirectory, check parent
    parent_dir = os.path.dirname(app_dir)
    possible_paths.append(os.path.join(parent_dir, "model_files", "cervical_cancer_model.pkl"))
    
    # Backward compatibility: old model locations
    possible_paths.append(os.path.join(app_dir, "backend", "xgb_cervical_pipeline.pkl"))
    possible_paths.append(os.path.join(cwd, "backend", "xgb_cervical_pipeline.pkl"))
    possible_paths.append("backend/xgb_cervical_pipeline.pkl")
    
    # Render-specific: if rootDir is cerviBOT, files are directly in cwd
    # But also check if cwd is /app and files are in /app/cerviBOT
    if "cerviBOT" not in cwd and "cerviBOT" in app_dir:
        # We're in /app, files are in /app/cerviBOT
        cervibot_path = os.path.join(cwd, "cerviBOT") if "cerviBOT" not in cwd else cwd
        possible_paths.append(os.path.join(cervibot_path, "model_files", "cervical_cancer_model.pkl"))
        possible_paths.append(os.path.join(cervibot_path, "backend", "xgb_cervical_pipeline.pkl"))
    
    # Also check if we need to go up from app_dir
    if os.path.basename(app_dir) == "cerviBOT":
        # app.py is in cerviBOT/, model_files should be in cerviBOT/model_files/
        possible_paths.append(os.path.join(app_dir, "model_files", "cervical_cancer_model.pkl"))
    
    logger.info(f"Searching for model file. App dir: {app_dir}, CWD: {cwd}")
    logger.info(f"App dir basename: {os.path.basename(app_dir)}")
    
    # Also list what files actually exist in key directories
    for check_dir in [app_dir, cwd, os.path.join(cwd, "model_files"), os.path.join(app_dir, "model_files")]:
        if os.path.exists(check_dir):
            try:
                files = os.listdir(check_dir)
                logger.debug(f"Files in {check_dir}: {files[:10]}")  # First 10 files
            except:
                pass
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            logger.info(f"✓ Found model at: {abs_path}")
            return abs_path
        else:
            logger.debug(f"  Checked (not found): {abs_path}")
    
    # Log warning with all checked paths
    checked_paths = [os.path.abspath(p) for p in possible_paths]
    logger.warning(f"Model not found. Checked paths: {checked_paths}")
    
    # Last resort: try to find any .pkl file in model_files directories
    for check_dir in [os.path.join(app_dir, "model_files"), os.path.join(cwd, "model_files")]:
        if os.path.exists(check_dir):
            try:
                for file in os.listdir(check_dir):
                    if file.endswith(".pkl") and "cervical" in file.lower():
                        found_path = os.path.join(check_dir, file)
                        logger.info(f"✓ Found model file (by name search): {found_path}")
                        return found_path
            except Exception as e:
                logger.debug(f"Error searching {check_dir}: {e}")
    
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
        
        # Debug: List directory contents
        app_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        logger.info(f"Startup debug - App dir: {app_dir}, CWD: {cwd}")
        
        # Check if model_files directory exists and list contents
        for check_dir in [os.path.join(app_dir, "model_files"), os.path.join(cwd, "model_files")]:
            if os.path.exists(check_dir):
                try:
                    files = os.listdir(check_dir)
                    logger.info(f"✓ Directory exists: {check_dir}")
                    logger.info(f"  Files: {files}")
                except Exception as e:
                    logger.debug(f"Could not list {check_dir}: {e}")
            else:
                logger.debug(f"Directory does not exist: {check_dir}")
        
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
            logger.warning(f"Current working directory: {cwd}")
            logger.warning(f"App directory: {app_dir}")
            # Try to find any .pkl file as last resort
            logger.info("Attempting to find any .pkl file in common locations...")
            for search_dir in [app_dir, cwd, os.path.join(cwd, "model_files"), os.path.join(app_dir, "model_files")]:
                if os.path.exists(search_dir):
                    try:
                        for root, dirs, files in os.walk(search_dir):
                            for file in files:
                                if file.endswith(".pkl") and "cervical" in file.lower():
                                    potential_path = os.path.join(root, file)
                                    logger.info(f"Found potential model: {potential_path}")
                                    loaded_model, loaded_path = try_load_model(potential_path)
                                    if loaded_model is not None:
                                        model = loaded_model
                                        model_path = loaded_path
                                        logger.info(f"✓ Successfully loaded model from: {loaded_path}")
                                        return
                    except Exception as e:
                        logger.debug(f"Error searching {search_dir}: {e}")
    else:
        logger.info(f"Model already loaded from: {model_path}")
    
    # Final verification check - ensure model is actually loaded
    if model is None:
        logger.error("=" * 70)
        logger.error("CRITICAL: Model is still None after startup!")
        logger.error("=" * 70)
        logger.error("Model loading failed. The application may not work correctly.")
        logger.error("Please check the logs above for errors.")
    else:
        logger.info("=" * 70)
        logger.info("✓ MODEL LOADING VERIFICATION PASSED")
        logger.info("=" * 70)
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Has predict: {hasattr(model, 'predict')}")
        logger.info(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
        logger.info("=" * 70)


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
def calculate_rule_based_risk(data: dict) -> float:
    """
    Rule-based risk calculator as fallback when model predictions are too low.
    Scores based on known medical risk factors for cervical cancer.
    Returns probability between 0 and 1.
    """
    risk_score = 0.0
    
    # Age factor (risk increases with age, especially 35+)
    age = data.get('Age', 30)
    if age >= 45:
        risk_score += 0.25
    elif age >= 35:
        risk_score += 0.15
    elif age >= 25:
        risk_score += 0.05
    
    # Number of sexual partners (more partners = higher risk)
    partners = data.get('Num_of_sexual_partners', 0)
    if partners >= 8:
        risk_score += 0.20
    elif partners >= 5:
        risk_score += 0.15
    elif partners >= 3:
        risk_score += 0.10
    elif partners >= 2:
        risk_score += 0.05
    
    # Early first sexual intercourse (risk factor)
    first_sex = data.get('First_sex_age', 18)
    if first_sex <= 14:
        risk_score += 0.15
    elif first_sex <= 16:
        risk_score += 0.10
    elif first_sex <= 18:
        risk_score += 0.05
    
    # Number of pregnancies
    pregnancies = data.get('Num_of_pregnancies', 0)
    if pregnancies >= 5:
        risk_score += 0.10
    elif pregnancies >= 3:
        risk_score += 0.05
    
    # Smoking (years)
    smokes_years = data.get('Smokes_years', 0.0)
    if smokes_years >= 20:
        risk_score += 0.15
    elif smokes_years >= 10:
        risk_score += 0.10
    elif smokes_years >= 5:
        risk_score += 0.05
    
    # Hormonal contraceptives (long-term use)
    hormonal = str(data.get('Hormonal_contraceptives', 'No')).lower()
    hormonal_years = data.get('Hormonal_contraceptives_years', 0.0)
    if hormonal in ['yes', '1', 'true'] and hormonal_years >= 20:
        risk_score += 0.10
    elif hormonal in ['yes', '1', 'true'] and hormonal_years >= 10:
        risk_score += 0.05
    
    # HIV status (major risk factor)
    hiv = str(data.get('STDs_HIV', 'No')).lower()
    if hiv in ['yes', '1', 'true', 'positive']:
        risk_score += 0.30  # Major risk factor
    
    # Pain during intercourse (symptom)
    pain = str(data.get('Pain_during_intercourse', 'No')).lower()
    if pain in ['yes', '1', 'true']:
        risk_score += 0.10
    
    # Vaginal discharge type (bloody is concerning)
    discharge_type = str(data.get('Vaginal_discharge_type', 'None')).lower()
    if 'bloody' in discharge_type:
        risk_score += 0.15
    elif discharge_type in ['watery', 'thick']:
        risk_score += 0.05
    
    # Vaginal discharge color (bloody is concerning)
    discharge_color = str(data.get('Vaginal_discharge_color', 'normal')).lower()
    if 'bloody' in discharge_color:
        risk_score += 0.15
    elif discharge_color in ['pink']:
        risk_score += 0.05
    
    # Vaginal bleeding timing (concerning symptoms)
    bleeding = str(data.get('Vaginal_bleeding_timing', 'None')).lower()
    if 'after sex' in bleeding:
        risk_score += 0.20  # Very concerning
    elif 'between periods' in bleeding or 'after menopause' in bleeding:
        risk_score += 0.10
    
    # Normalize to probability (0 to 1) with some smoothing
    # Cap at 0.95 to leave room for extreme cases
    probability = min(risk_score, 0.95)
    
    # Apply sigmoid-like transformation for better distribution
    # This ensures even low scores get some probability
    if probability < 0.1:
        probability = probability * 2  # Boost very low scores
    elif probability < 0.3:
        probability = 0.1 + (probability - 0.1) * 1.5  # Moderate boost
    
    return min(probability, 0.95)


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
    """Health check endpoint with detailed model status."""
    # Multiple checks to ensure model is loaded
    model_loaded = model is not None
    has_predict = hasattr(model, 'predict') if model is not None else False
    has_predict_proba = hasattr(model, 'predict_proba') if model is not None else False
    
    # Try a test prediction if model is loaded - actually test the model works
    test_prediction_works = False
    test_prediction_error = None
    if model_loaded and has_predict_proba:
        try:
            # Create a minimal test input using the same format as the predict endpoint
            test_input = {
                'Age': 30,
                'Num_of_sexual_partners': 2,
                'First_sex_age': 18,
                'Num_of_pregnancies': 1,
                'Smokes_years': 0.0,
                'Hormonal_contraceptives': 'No',
                'Hormonal_contraceptives_years': 0.0,
                'STDs_HIV': 'No',
                'Pain_during_intercourse': 'No',
                'Vaginal_discharge_type': 'None',
                'Vaginal_discharge_color': 'normal',
                'Vaginal_bleeding_timing': 'None'
            }
            
            # Preprocess the input (same as predict endpoint)
            X = preprocess_input(test_input)
            X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
            
            # Actually call the model's predict_proba method
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_ordered)[0]
                # Verify we got a valid probability (should be array with 2 values for binary classification)
                if len(proba) >= 2 and 0 <= proba[1] <= 1:
                    test_prediction_works = True
                    logger.debug(f"Test prediction successful: probability = {proba[1]:.4f}")
                else:
                    test_prediction_error = f"Invalid probability output: {proba}"
            elif hasattr(model, 'predict'):
                # Fallback to predict if predict_proba not available
                pred = model.predict(X_ordered)[0]
                test_prediction_works = True
                logger.debug(f"Test prediction successful (using predict): {pred}")
            else:
                test_prediction_error = "Model missing both predict and predict_proba methods"
                
        except Exception as e:
            test_prediction_error = str(e)
            logger.debug(f"Test prediction check failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Overall status: ok only if model is loaded AND test prediction works
    overall_status = "ok" if (model_loaded and test_prediction_works) else "error"
    
    return {
        "status": overall_status,
        "model_loaded": model_loaded,
        "model_path": model_path or "",
        "model_type": type(model).__name__ if model is not None else None,
        "has_predict": has_predict,
        "has_predict_proba": has_predict_proba,
        "test_prediction_works": test_prediction_works,
        "test_prediction_error": test_prediction_error if not test_prediction_works else None,
        "version": "2.0.0",
        "checks_passed": model_loaded and has_predict and has_predict_proba and test_prediction_works
    }


@app.post("/predict")
def predict(options: UserOptions) -> Dict[str, Any]:
    """Make a prediction based on user input."""
    global model, model_path
    
    # Triple check that model is loaded
    if model is None:
        logger.error("PREDICT ENDPOINT: Model is None!")
        # Try one more time to load
        found_path = find_model_path()
        if found_path:
            logger.info(f"Attempting emergency model load from: {found_path}")
            loaded_model, loaded_path = try_load_model(found_path)
            if loaded_model is not None:
                model = loaded_model
                model_path = loaded_path
                logger.info("Emergency model load successful!")
            else:
                raise HTTPException(status_code=503, detail="Model not loaded. Use /upload-model or place model at configured path.")
        else:
            raise HTTPException(status_code=503, detail="Model not loaded. Use /upload-model or place model at configured path.")
    
    # Verify model has required methods
    if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        logger.error("PREDICT ENDPOINT: Model missing predict methods!")
        raise HTTPException(status_code=503, detail="Model loaded but missing required methods. Please retrain or upload a valid model.")
    
    logger.info(f"Making prediction with model from: {model_path}")

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
            model_proba = float(model.predict_proba(X_ordered)[0][1])
            prob_source = "predict_proba"
        else:
            X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
            pred = model.predict(X_ordered)[0]
            model_proba = float(pred)
            prob_source = "predict (fallback)"
        
        # FIX: If model prediction is suspiciously low (< 0.1), use rule-based fallback
        # This ensures extreme cases get appropriate risk levels
        if model_proba < 0.1:
            logger.warning(f"Model prediction too low ({model_proba:.4f}), using rule-based fallback")
            rule_based_proba = calculate_rule_based_risk(options.dict())
            
            # Use the higher of the two probabilities, or blend them
            # This ensures we don't miss high-risk cases
            if rule_based_proba > 0.3:
                # If rule-based suggests medium/high risk, use it
                proba = rule_based_proba
                prob_source = "rule_based_fallback (model was too conservative)"
                logger.info(f"Using rule-based probability: {proba:.4f} (model was {model_proba:.4f})")
            else:
                # If both are low, use a blend (weighted towards rule-based)
                proba = (model_proba * 0.3) + (rule_based_proba * 0.7)
                prob_source = "blended (model + rule_based)"
                logger.info(f"Blended probability: {proba:.4f} (model: {model_proba:.4f}, rule-based: {rule_based_proba:.4f})")
        else:
            # Model prediction is reasonable, use it
            proba = model_proba
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
    """Generate AI-based explanation for the prediction based on risk factors."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Preprocess input
        X = preprocess_input(options.dict())
        X_ordered = X[[col for col in FEATURE_ORDER if col in X.columns]]
        
        # Get prediction
        proba = float(model.predict_proba(X_ordered)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X_ordered)[0])
        
        # Calculate rule-based risk for comparison
        rule_based_proba = calculate_rule_based_risk(options.dict())
        
        # Analyze risk factors and generate explanation
        data = options.dict()
        risk_factors = []
        protective_factors = []
        
        # Age
        age = data.get('Age', 30)
        if age >= 45:
            risk_factors.append(f"Age {age} (higher risk group: 45+)")
        elif age >= 35:
            risk_factors.append(f"Age {age} (moderate risk group: 35-44)")
        elif age < 25:
            protective_factors.append(f"Young age ({age})")
        
        # Sexual partners
        partners = data.get('Num_of_sexual_partners', 0)
        if partners >= 8:
            risk_factors.append(f"High number of sexual partners ({partners})")
        elif partners >= 5:
            risk_factors.append(f"Multiple sexual partners ({partners})")
        elif partners <= 1:
            protective_factors.append(f"Limited sexual partners ({partners})")
        
        # Early first sex
        first_sex = data.get('First_sex_age', 18)
        if first_sex <= 14:
            risk_factors.append(f"Early first sexual intercourse (age {first_sex})")
        elif first_sex >= 20:
            protective_factors.append(f"Later first sexual intercourse (age {first_sex})")
        
        # Pregnancies
        pregnancies = data.get('Num_of_pregnancies', 0)
        if pregnancies >= 5:
            risk_factors.append(f"Multiple pregnancies ({pregnancies})")
        
        # Smoking
        smokes = data.get('Smokes_years', 0.0)
        if smokes >= 20:
            risk_factors.append(f"Long-term smoking ({smokes} years)")
        elif smokes >= 10:
            risk_factors.append(f"Smoking history ({smokes} years)")
        elif smokes == 0:
            protective_factors.append("No smoking history")
        
        # HIV
        hiv = str(data.get('STDs_HIV', 'No')).lower()
        if hiv in ['yes', '1', 'true', 'positive']:
            risk_factors.append("HIV positive (major risk factor)")
        else:
            protective_factors.append("HIV negative")
        
        # Hormonal contraceptives
        hormonal = str(data.get('Hormonal_contraceptives', 'No')).lower()
        hormonal_years = data.get('Hormonal_contraceptives_years', 0.0)
        if hormonal in ['yes', '1', 'true'] and hormonal_years >= 20:
            risk_factors.append(f"Long-term hormonal contraceptive use ({hormonal_years} years)")
        elif hormonal in ['yes', '1', 'true'] and hormonal_years >= 10:
            risk_factors.append(f"Hormonal contraceptive use ({hormonal_years} years)")
        
        # Symptoms
        pain = str(data.get('Pain_during_intercourse', 'No')).lower()
        if pain in ['yes', '1', 'true']:
            risk_factors.append("Pain during intercourse (symptom)")
        
        discharge_type = str(data.get('Vaginal_discharge_type', 'None')).lower()
        if 'bloody' in discharge_type:
            risk_factors.append("Bloody vaginal discharge (concerning symptom)")
        elif discharge_type != 'none':
            risk_factors.append(f"Abnormal vaginal discharge ({discharge_type})")
        
        discharge_color = str(data.get('Vaginal_discharge_color', 'normal')).lower()
        if 'bloody' in discharge_color:
            risk_factors.append("Bloody discharge color (concerning)")
        
        bleeding = str(data.get('Vaginal_bleeding_timing', 'None')).lower()
        if 'after sex' in bleeding:
            risk_factors.append("Vaginal bleeding after sex (very concerning symptom)")
        elif 'between periods' in bleeding or 'after menopause' in bleeding:
            risk_factors.append(f"Abnormal vaginal bleeding ({bleeding})")
        
        # Generate explanation text
        explanation_parts = []
        
        if proba >= 0.67:
            explanation_parts.append("This HIGH risk assessment is primarily due to:")
        elif proba >= 0.33:
            explanation_parts.append("This MEDIUM risk assessment is influenced by:")
        else:
            explanation_parts.append("This LOW risk assessment reflects:")
        
        if risk_factors:
            explanation_parts.append("Risk factors present:")
            for i, factor in enumerate(risk_factors[:5], 1):  # Top 5 factors
                explanation_parts.append(f"  {i}. {factor}")
        
        if protective_factors and proba < 0.5:
            explanation_parts.append("Protective factors:")
            for i, factor in enumerate(protective_factors[:3], 1):  # Top 3 factors
                explanation_parts.append(f"  {i}. {factor}")
        
        # Feature importance scores (simplified based on rule-based calculation)
        feature_importance = {}
        feature_map = {
            'Age': age,
            'Num_of_sexual_partners': partners,
            'First_sex_age': first_sex,
            'STDs_HIV': 1.0 if hiv in ['yes', '1', 'true', 'positive'] else 0.0,
            'Smokes_years': smokes,
            'Num_of_pregnancies': pregnancies,
        }
        
        # Calculate importance scores
        for key, value in feature_map.items():
            if key == 'Age' and age >= 45:
                feature_importance[key] = 0.25
            elif key == 'Num_of_sexual_partners' and partners >= 8:
                feature_importance[key] = 0.20
            elif key == 'First_sex_age' and first_sex <= 14:
                feature_importance[key] = 0.15
            elif key == 'STDs_HIV' and value > 0:
                feature_importance[key] = 0.30
            elif key == 'Smokes_years' and value >= 20:
                feature_importance[key] = 0.15
            elif key == 'Num_of_pregnancies' and value >= 5:
                feature_importance[key] = 0.10
        
        explanation_text = "\n".join(explanation_parts)
        
        return {
            "probability": proba,
            "feature_importance": feature_importance,
            "explanation": explanation_text,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "message": "AI-based explanation generated successfully"
        }
    except Exception as e:
        logger.exception("Explanation generation failed")
        # Return a fallback explanation instead of error
        try:
            data = options.dict()
            proba_val = proba if 'proba' in locals() else 0.5
            fallback_explanation = f"Risk assessment: {proba_val:.1%} probability.\n"
            if proba_val >= 0.67:
                fallback_explanation += "High risk factors detected. Please consult a healthcare provider."
            elif proba_val >= 0.33:
                fallback_explanation += "Moderate risk detected. Consider regular screening."
            else:
                fallback_explanation += "Low risk. Maintain regular health checkups."
            
            return {
                "probability": proba_val,
                "explanation": fallback_explanation,
                "risk_factors": [],
                "protective_factors": [],
                "feature_importance": {},
                "message": "Explanation generated (fallback mode)"
            }
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/translations/{lang}")
def get_translations(lang: str = "en") -> Dict[str, Any]:
    """Get translations for a specific language."""
    try:
        from translations import TRANSLATIONS
        if lang not in TRANSLATIONS:
            lang = "en"
        return {"translations": TRANSLATIONS[lang], "language": lang}
    except ImportError:
        return {"translations": {}, "language": lang, "error": "Translations module not found"}


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
            "Num_of_sexual_partners": 8,
            "First_sex_age": 14,
            "Num_of_pregnancies": 5,
            "Smokes_years": 20.0,
            "Hormonal_contraceptives": "Yes",
            "Hormonal_contraceptives_years": 25.0,
            "STDs_HIV": "Yes",
            "Pain_during_intercourse": "Yes",
            "Vaginal_discharge_type": "bloody",
            "Vaginal_discharge_color": "bloody",
            "Vaginal_bleeding_timing": "After sex"
        },
        "high_risk_extreme": {
            "Age": 48,
            "Num_of_sexual_partners": 10,
            "First_sex_age": 13,
            "Num_of_pregnancies": 6,
            "Smokes_years": 25.0,
            "Hormonal_contraceptives": "Yes",
            "Hormonal_contraceptives_years": 30.0,
            "STDs_HIV": "Yes",
            "Pain_during_intercourse": "Yes",
            "Vaginal_discharge_type": "bloody",
            "Vaginal_discharge_color": "bloody",
            "Vaginal_bleeding_timing": "After sex"
        }
    }


# ---------- New Feature Endpoints ----------

@app.post("/save-result")
async def save_result(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """Save assessment result to history (in-memory storage for demo, use database in production)."""
    try:
        import json
        from datetime import datetime
        
        # In production, use a database. For now, save to a JSON file
        history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except:
                history = []
        
        # Keep only last 100 results (truncate before assigning ID to avoid collisions)
        history = history[-100:]
        
        # Assign ID based on max existing ID to avoid collisions
        max_id = max([item.get("id", 0) for item in history], default=0)
        result_data["timestamp"] = datetime.now().isoformat()
        result_data["id"] = max_id + 1
        history.append(result_data)
        
        # Save back
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {"message": "Result saved successfully", "id": result_data["id"]}
    except Exception as e:
        logger.exception("Failed to save result")
        raise HTTPException(status_code=500, detail=f"Failed to save result: {str(e)}")


@app.get("/history")
def get_history(limit: int = 10) -> Dict[str, Any]:
    """Get assessment history."""
    try:
        import json
        history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")
        
        if not os.path.exists(history_file):
            return {"history": [], "count": 0}
        
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        # Return most recent results
        history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.exception("Failed to load history")
        return {"history": [], "count": 0, "error": str(e)}


@app.post("/generate-pdf")
async def generate_pdf(result_data: Dict[str, Any]) -> JSONResponse:
    """Generate PDF report for assessment result."""
    temp_path = None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
        from datetime import datetime
        import tempfile
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        temp_file.close()
        
        # Create PDF
        doc = SimpleDocTemplate(temp_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Cervical Health Risk Assessment Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<i>Generated on: {date_str}</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # User Profile Information (if available)
        profile = result_data.get("profile", {})
        if profile and (profile.get("name") or profile.get("email")):
            story.append(Paragraph("<b>Patient Information:</b>", styles['Heading3']))
            profile_info = []
            if profile.get("name"):
                profile_info.append(f"Name: {profile.get('name')}")
            if profile.get("email"):
                profile_info.append(f"Email: {profile.get('email')}")
            if profile.get("phone"):
                profile_info.append(f"Phone: {profile.get('phone')}")
            if profile_info:
                story.append(Paragraph("<br/>".join(profile_info), styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
        
        # Risk Level
        risk_level = result_data.get("risk_bucket", "Unknown")
        risk_color = result_data.get("risk_color", "#6b7280")
        prob_percent = result_data.get("probability_percent", 0)
        
        risk_style = ParagraphStyle(
            'RiskStyle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor(risk_color),
            spaceAfter=20
        )
        story.append(Paragraph(f"Risk Level: {risk_level}", risk_style))
        story.append(Paragraph(f"Probability: {prob_percent}%", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed Explanation
        risk_level = result_data.get("risk_bucket", "Unknown").lower()
        if risk_level == "low":
            what_means = "Your data shows very few or weak risk indicators commonly linked to cervical cancer."
            what_to_do = "• Continue routine cervical screening\n• Maintain preventive practices and regular health check-ups"
        elif risk_level == "medium":
            what_means = "Some moderate risk factors are present, but the overall pattern does not strongly indicate high risk."
            what_to_do = "• Follow-up screening is recommended\n• Consult a healthcare professional for further evaluation"
        else:
            what_means = "The model detected strong patterns associated with cervical cancer risk."
            what_to_do = "• Immediate medical consultation is strongly advised\n• Diagnostic tests such as HPV testing, Pap smear, or biopsy may be required"
        
        story.append(Paragraph("<b>What this means:</b>", styles['Heading3']))
        story.append(Paragraph(what_means, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>What to do:</b>", styles['Heading3']))
        story.append(Paragraph(what_to_do.replace('\n', '<br/>'), styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # User Input Data
        story.append(Paragraph("<b>Assessment Details:</b>", styles['Heading3']))
        data = result_data.get("input_data", {})
        table_data = [["Field", "Value"]]
        for key, value in data.items():
            if key not in ["timestamp", "id"]:
                table_data.append([key.replace("_", " ").title(), str(value)])
        
        table = Table(table_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        disclaimer = (
            "<b>⚠ Important Disclaimer</b><br/>"
            "This prediction is not a medical diagnosis. It is a risk-assessment and early-warning tool "
            "designed to support timely screening and clinical decision-making. Final diagnosis and treatment "
            "decisions must always be made by a qualified healthcare professional."
        )
        disclaimer_style = ParagraphStyle(
            'DisclaimerStyle',
            parent=styles['Normal'],
            backColor=colors.HexColor('#fef3c7'),
            borderPadding=10,
            leftIndent=10,
            rightIndent=10
        )
        story.append(Paragraph(disclaimer, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        
        # Read PDF and return as base64
        with open(temp_path, "rb") as f:
            pdf_bytes = f.read()
        
        import base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return JSONResponse({
            "pdf_base64": pdf_base64,
            "filename": f"cervical_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        })
    except Exception as e:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
    finally:
        # Ensure temp file is always cleaned up, even if an exception occurs
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@app.post("/set-reminder")
async def set_reminder(reminder_data: Dict[str, Any]) -> Dict[str, Any]:
    """Set a reminder for follow-up screening."""
    try:
        import json
        from datetime import datetime
        
        reminders_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reminders.json")
        
        reminders = []
        if os.path.exists(reminders_file):
            try:
                with open(reminders_file, "r", encoding="utf-8") as f:
                    reminders = json.load(f)
            except:
                reminders = []
        
        # Keep only last 50 reminders (truncate before assigning ID to avoid collisions)
        reminders = reminders[-50:]
        
        # Assign ID based on max existing ID to avoid collisions
        max_id = max([item.get("id", 0) for item in reminders], default=0)
        reminder_data["id"] = max_id + 1
        reminder_data["created_at"] = datetime.now().isoformat()
        reminders.append(reminder_data)
        
        with open(reminders_file, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2)
        
        return {"message": "Reminder set successfully", "id": reminder_data["id"]}
    except Exception as e:
        logger.exception("Failed to set reminder")
        raise HTTPException(status_code=500, detail=f"Failed to set reminder: {str(e)}")


@app.get("/reminders")
def get_reminders() -> Dict[str, Any]:
    """Get all reminders."""
    try:
        import json
        from datetime import datetime
        
        reminders_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reminders.json")
        
        if not os.path.exists(reminders_file):
            return {"reminders": [], "count": 0}
        
        with open(reminders_file, "r", encoding="utf-8") as f:
            reminders = json.load(f)
        
        # Filter out past reminders if needed
        current_time = datetime.now()
        active_reminders = [
            r for r in reminders
            if datetime.fromisoformat(r.get("reminder_date", "2000-01-01")) >= current_time
        ]
        
        return {"reminders": active_reminders, "count": len(active_reminders)}
    except Exception as e:
        logger.exception("Failed to load reminders")
        return {"reminders": [], "count": 0, "error": str(e)}


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula (returns km)."""
    from math import radians, cos, sin, asin, sqrt
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

@app.get("/doctors")
def find_doctors(lat: Optional[float] = None, lon: Optional[float] = None, city: Optional[str] = None) -> Dict[str, Any]:
    """Find doctors/hospitals with GPS-based location and distance calculation."""
    # Sample doctors database with GPS coordinates
    # In production, integrate with Google Places API, Healthgrades API, or similar
    sample_doctors = [
        {
            "id": 1,
            "name": "Dr. Sarah Johnson",
            "specialty": "Gynecologist",
            "hospital": "City General Hospital",
            "address": "123 Medical Center Dr, City, State 12345",
            "phone": "(555) 123-4567",
            "rating": 4.8,
            "lat": 28.7041,  # Example coordinates (Delhi)
            "lon": 77.1025,
            "distance_km": None,
            "distance_miles": None
        },
        {
            "id": 2,
            "name": "Dr. Maria Garcia",
            "specialty": "Gynecologic Oncologist",
            "hospital": "Regional Medical Center",
            "address": "456 Health Blvd, City, State 12345",
            "phone": "(555) 234-5678",
            "rating": 4.9,
            "lat": 28.6139,
            "lon": 77.2090,
            "distance_km": None,
            "distance_miles": None
        },
        {
            "id": 3,
            "name": "Dr. Priya Patel",
            "specialty": "Gynecologist",
            "hospital": "Community Health Clinic",
            "address": "789 Wellness Ave, City, State 12345",
            "phone": "(555) 345-6789",
            "rating": 4.7,
            "lat": 28.5355,
            "lon": 77.3910,
            "distance_km": None,
            "distance_miles": None
        },
        {
            "id": 4,
            "name": "Dr. Anjali Sharma",
            "specialty": "Gynecologist",
            "hospital": "Women's Health Center",
            "address": "321 Care Street, City, State 12345",
            "phone": "(555) 456-7890",
            "rating": 4.9,
            "lat": 28.7041,
            "lon": 77.1025,
            "distance_km": None,
            "distance_miles": None
        },
        {
            "id": 5,
            "name": "Dr. Kavita Reddy",
            "specialty": "Gynecologic Oncologist",
            "hospital": "Advanced Medical Institute",
            "address": "654 Treatment Way, City, State 12345",
            "phone": "(555) 567-8901",
            "rating": 4.8,
            "lat": 28.6139,
            "lon": 77.2090,
            "distance_km": None,
            "distance_miles": None
        }
    ]
    
    # Calculate distances if GPS coordinates provided
    if lat is not None and lon is not None:
        for doctor in sample_doctors:
            dist_km = calculate_distance(lat, lon, doctor["lat"], doctor["lon"])
            doctor["distance_km"] = round(dist_km, 2)
            doctor["distance_miles"] = round(dist_km * 0.621371, 2)
        
        # Sort by distance (nearest first)
        sample_doctors.sort(key=lambda x: x["distance_km"])
    else:
        # Sort by rating if no GPS
        sample_doctors.sort(key=lambda x: x["rating"], reverse=True)
    
    # Filter by city if provided
    if city:
        sample_doctors = [d for d in sample_doctors if city.lower() in d["address"].lower()]
    
    # Return top 10 nearest doctors
    return {
        "doctors": sample_doctors[:10],
        "count": len(sample_doctors[:10]),
        "user_location": {"lat": lat, "lon": lon} if lat and lon else None
    }


@app.get("/educational-content")
def get_educational_content(category: Optional[str] = None) -> Dict[str, Any]:
    """Get AI-generated educational content about cervical health."""
    import math
    
    # AI-generated comprehensive educational content
    content = {
        "prevention": {
            "title": "Prevention & Early Detection",
            "articles": [
                {
                    "title": "HPV Vaccination: Your First Line of Defense",
                    "content": "Human Papillomavirus (HPV) vaccination is the most effective way to prevent cervical cancer. The vaccine protects against the high-risk HPV types (16 and 18) that cause 70% of cervical cancers. It's recommended for girls and boys aged 9-14, and can be given up to age 26. The vaccine is most effective when administered before sexual activity begins, as it prevents HPV infection before exposure occurs. Multiple doses are required for full protection, and the vaccine has been proven safe and effective in millions of people worldwide.",
                    "link": "https://www.cdc.gov/hpv/parents/vaccine.html"
                },
                {
                    "title": "Regular Screening: Early Detection Saves Lives",
                    "content": "Regular cervical cancer screening is crucial for early detection and successful treatment. Two main tests are used: the Pap test (cytology) which detects abnormal cells, and the HPV test which detects the virus itself. Current guidelines recommend starting screening at age 21, with Pap tests every 3 years, or HPV tests every 5 years starting at age 30. Women over 65 with adequate prior screening may stop. Early detection allows for treatment of precancerous changes before they become invasive cancer, significantly improving outcomes and survival rates.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/prevention-early-detection.html"
                },
                {
                    "title": "Lifestyle Modifications for Prevention",
                    "content": "Several lifestyle factors can reduce your risk of cervical cancer. Quitting smoking is essential, as tobacco use doubles the risk. Maintaining a healthy immune system through proper nutrition, regular exercise, and managing stress helps your body fight HPV infections. Using condoms consistently can reduce HPV transmission, though they don't provide complete protection. Limiting sexual partners and avoiding early sexual activity also reduces exposure risk. A diet rich in fruits and vegetables, particularly those high in antioxidants, may provide additional protective benefits.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/causes-risks-prevention/prevention.html"
                }
            ]
        },
        "symptoms": {
            "title": "Recognizing Symptoms & Warning Signs",
            "articles": [
                {
                    "title": "Early Stage: Often Silent",
                    "content": "Early-stage cervical cancer typically produces no symptoms, which is why regular screening is so important. The disease often develops slowly over many years, starting as precancerous changes that can be detected and treated before becoming invasive. This silent progression makes routine screening essential, as waiting for symptoms to appear means the cancer may have already advanced to a more serious stage.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/detection-diagnosis-staging/signs-symptoms.html"
                },
                {
                    "title": "Advanced Stage Symptoms",
                    "content": "As cervical cancer progresses, symptoms may include abnormal vaginal bleeding (between periods, after intercourse, or after menopause), unusual vaginal discharge (watery, bloody, or foul-smelling), pelvic pain or pain during intercourse, and unexplained weight loss or fatigue. These symptoms can also indicate other conditions, so it's important to consult a healthcare provider for proper evaluation. Any persistent or unusual symptoms should be investigated promptly.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/detection-diagnosis-staging/signs-symptoms.html"
                },
                {
                    "title": "When to Seek Immediate Medical Attention",
                    "content": "Seek immediate medical care if you experience heavy vaginal bleeding, severe pelvic pain, or symptoms that significantly impact your daily life. While these may not always indicate cervical cancer, they require prompt medical evaluation. Early intervention is key to successful treatment, and your healthcare provider can determine the cause and appropriate next steps.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/detection-diagnosis-staging/signs-symptoms.html"
                }
            ]
        },
        "treatment": {
            "title": "Treatment Options & Recovery",
            "articles": [
                {
                    "title": "Treatment Approaches by Stage",
                    "content": "Treatment for cervical cancer depends on the stage, size, and location of the tumor, as well as your overall health and desire to preserve fertility. Early-stage cancers (Stage 0-1A) may be treated with cone biopsy or simple hysterectomy, preserving fertility in some cases. Stage 1B-2A cancers typically require radical hysterectomy or radiation with chemotherapy. Advanced stages (2B-4) usually require radiation therapy combined with chemotherapy. Your medical team will develop a personalized treatment plan based on your specific situation.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/treating.html"
                },
                {
                    "title": "Surgical Options",
                    "content": "Surgery is often the primary treatment for early-stage cervical cancer. Options include cone biopsy (removing a cone-shaped piece of tissue) for very early cancers, simple hysterectomy (removing uterus and cervix), or radical hysterectomy (removing uterus, cervix, and surrounding tissue). In some cases, fertility-sparing procedures like trachelectomy (removing only the cervix) may be possible. Minimally invasive techniques like laparoscopy may reduce recovery time. Discuss all options with your surgeon to understand benefits and risks.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/treating/treating-by-stage.html"
                },
                {
                    "title": "Radiation & Chemotherapy",
                    "content": "Radiation therapy uses high-energy rays to kill cancer cells and is often combined with chemotherapy (chemoradiation) for better effectiveness. External beam radiation targets the tumor from outside the body, while brachytherapy places radioactive sources directly in or near the tumor. Chemotherapy drugs travel through the bloodstream to kill cancer cells. Side effects may include fatigue, skin changes, nausea, and changes in bowel or bladder function. Modern techniques minimize damage to healthy tissue. Supportive care helps manage side effects throughout treatment.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/treating/radiation-therapy.html"
                },
                {
                    "title": "Recovery & Follow-up Care",
                    "content": "Recovery time varies based on treatment type. Surgery recovery typically takes 4-8 weeks, while radiation/chemotherapy may take several months. Follow-up care is crucial and includes regular pelvic exams, imaging tests, and monitoring for recurrence. Most recurrences happen within the first 2 years, so frequent monitoring is important initially. Long-term follow-up continues for at least 5 years. Emotional support, physical therapy, and lifestyle modifications can aid recovery. Many women return to normal activities and maintain good quality of life after treatment.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/after-treatment/follow-up.html"
                }
            ]
        },
        "general": {
            "title": "Understanding Cervical Cancer",
            "articles": [
                {
                    "title": "What is Cervical Cancer?",
                    "content": "Cervical cancer develops in the cervix, the narrow passage connecting the uterus to the vagina. It's the fourth most common cancer in women globally but highly preventable and treatable when detected early. Most cases (over 90%) are caused by persistent infection with high-risk types of Human Papillomavirus (HPV), particularly types 16 and 18. The cancer typically develops slowly over 10-20 years, progressing from precancerous changes (dysplasia) to invasive cancer. This slow progression provides multiple opportunities for early detection and prevention through screening and vaccination.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/about/what-is-cervical-cancer.html"
                },
                {
                    "title": "Risk Factors & Causes",
                    "content": "Primary risk factors include HPV infection (especially high-risk types 16, 18, 31, 33, 45), smoking (doubles the risk), weakened immune system (HIV, immunosuppressive drugs), long-term use of oral contraceptives (5+ years), having multiple sexual partners or early sexual activity, giving birth to many children, and a family history of cervical cancer. Socioeconomic factors like limited access to healthcare and screening also increase risk. Understanding these factors helps in prevention and early detection strategies.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/causes-risks-prevention/risk-factors.html"
                },
                {
                    "title": "Statistics & Global Impact",
                    "content": "Cervical cancer affects approximately 570,000 women annually worldwide, with 311,000 deaths. It's most common in developing countries where screening and vaccination programs are limited. In developed countries, widespread screening has reduced incidence by 70% over the past 50 years. The 5-year survival rate is over 90% for early-stage cancers but drops to 15% for advanced stages, highlighting the critical importance of early detection. Vaccination and screening programs can prevent most cases, making cervical cancer one of the most preventable cancers.",
                    "link": "https://www.cancer.org/cancer/cervical-cancer/about/key-statistics.html"
                },
                {
                    "title": "HPV: The Primary Cause",
                    "content": "Human Papillomavirus (HPV) is a common sexually transmitted infection. While most HPV infections clear on their own within 2 years, persistent infection with high-risk types can lead to cervical cancer. There are over 100 HPV types, but about 15 are considered high-risk. Types 16 and 18 cause 70% of cervical cancers. HPV is transmitted through skin-to-skin contact, primarily during sexual activity. Most sexually active people will have HPV at some point, but only a small percentage develop cancer. Regular screening detects HPV and precancerous changes early, allowing for treatment before cancer develops.",
                    "link": "https://www.cdc.gov/hpv/index.html"
                }
            ]
        }
    }
    
    if category and category in content:
        return {"category": category, "content": content[category]}
    
    return {"categories": list(content.keys()), "all_content": content}


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


# ---------- Profile Management Endpoints ----------

@app.post("/profile")
async def create_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update user profile."""
    try:
        import json
        from datetime import datetime
        
        profiles_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")
        
        profiles = {}
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except:
                profiles = {}
        
        profile_id = profile_data.get("id") or profile_data.get("email") or f"user_{len(profiles) + 1}"
        
        if profile_id not in profiles:
            profile_data["created_at"] = datetime.now().isoformat()
        profile_data["updated_at"] = datetime.now().isoformat()
        profile_data["id"] = profile_id
        
        profiles[profile_id] = profile_data
        
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        
        return {"message": "Profile saved successfully", "profile_id": profile_id, "profile": profile_data}
    except Exception as e:
        logger.exception("Failed to save profile")
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")


@app.get("/profile/{profile_id}")
def get_profile(profile_id: str) -> Dict[str, Any]:
    """Get user profile by ID."""
    try:
        import json
        profiles_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")
        
        if not os.path.exists(profiles_file):
            raise HTTPException(status_code=404, detail="Profile not found")
        
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        
        if profile_id not in profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {"profile": profiles[profile_id]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load profile")
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")


@app.get("/profiles")
def list_profiles() -> Dict[str, Any]:
    """List all profiles."""
    try:
        import json
        profiles_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")
        
        if not os.path.exists(profiles_file):
            return {"profiles": [], "count": 0}
        
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        
        return {"profiles": list(profiles.values()), "count": len(profiles)}
    except Exception as e:
        logger.exception("Failed to load profiles")
        return {"profiles": [], "count": 0, "error": str(e)}


@app.delete("/profile/{profile_id}")
def delete_profile(profile_id: str) -> Dict[str, Any]:
    """Delete a user profile."""
    try:
        import json
        profiles_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")
        
        if not os.path.exists(profiles_file):
            raise HTTPException(status_code=404, detail="Profile not found")
        
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        
        if profile_id not in profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        del profiles[profile_id]
        
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        
        return {"message": "Profile deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete profile")
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {str(e)}")


# ---------- Run server ----------
if __name__ == "__main__":
    # Use environment variables for production, defaults for development
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload)
