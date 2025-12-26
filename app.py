# app.py
import os
import logging
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# ---------- Configuration ----------
MODEL_PATH = r"C:\Users\Satyam Raj\cerviBOT\backend\xgb_cervical_pipeline.pkl"

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

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

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cervi_backend")

# ---------- App & CORS ----------
app = FastAPI(title="Cervical Cancer Risk Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model holder ----------
model = None
model_path = None

# ---------- WhatsApp Conversation State ----------
user_sessions = {}  # Store user conversation state by phone number


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


# Try load at startup
if os.path.exists(MODEL_PATH):
    model, model_path = try_load_model(MODEL_PATH)
else:
    logger.info(f"Model file not found at {MODEL_PATH}. Use /upload-model to upload one.")


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


# ---------- WhatsApp Integration ----------
def send_whatsapp_message_via_api(to_number: str, message: str) -> bool:
    """
    Send WhatsApp message using Twilio REST API (alternative to TwiML response).
    Use this when you need to send messages outside of webhook response.
    """
    try:
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.error("Twilio credentials not configured")
            return False
            
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Handle long messages by splitting
        messages_to_send = []
        if len(message) > 1600:  # WhatsApp recommended limit
            parts = [message[i:i+1600] for i in range(0, len(message), 1600)]
            messages_to_send = parts
        else:
            messages_to_send = [message]
        
        for i, msg_part in enumerate(messages_to_send):
            if i > 0:
                msg_part = f"(cont.) {msg_part}"
                
            twilio_message = client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                body=msg_part,
                to=to_number
            )
            logger.info(f"Sent WhatsApp message via API: {twilio_message.sid}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message via API: {e}")
        return False


def process_chatbot_message(message: str, user_id: str) -> str:
    """
    Process incoming message and return chatbot response.
    Calls the existing cervicare_chatbot function.
    """
    try:
        # Store conversation state if needed
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                'message_count': 0,
                'last_message': None,
                'conversation_start': datetime.now().isoformat()
            }
        
        # Update session state
        user_sessions[user_id]['message_count'] += 1
        user_sessions[user_id]['last_message'] = message
        
        # Call your existing chatbot function
        response_text = cervicare_chatbot(message, user_id)
        return response_text
    except Exception as e:
        logger.error(f"Error in chatbot processing: {e}")
        return "Sorry, I encountered an error. Please try again later."


# ---------- Endpoints ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path or "",
        "active_sessions": len(user_sessions),
    }


@app.get("/sessions")
def get_sessions() -> Dict[str, Any]:
    """View current WhatsApp sessions (for debugging)"""
    return {
        "active_sessions": len(user_sessions),
        "sessions": user_sessions
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


# ---------- WhatsApp Webhook ----------
@app.post("/whatsapp/webhook", response_class=PlainTextResponse)
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
    To: str = Form(...),
    MessageSid: str = Form(...),
    AccountSid: str = Form(...),
    MessagingServiceSid: str = Form(None),
    NumMedia: str = Form("0"),
    MediaContentType: str = Form(None),
    MediaUrl: str = Form(None)
):
    """
    Complete WhatsApp webhook endpoint for Twilio.
    Returns valid TwiML response for immediate message delivery.
    """
    try:
        # Validate required parameters
        if not From or not Body or not To or not MessageSid:
            logger.error("Missing required Twilio parameters")
            return _create_error_twiml("Missing required parameters")
        
        # Extract user phone number and message
        user_phone = From.strip()
        message_text = Body.strip()
        to_number = To.strip()
        message_sid = MessageSid.strip()
        
        # Validate phone number format
        if not user_phone.startswith('whatsapp:+'):
            logger.error(f"Invalid From format: {user_phone}")
            return _create_error_twiml("Invalid phone number format")
        
        logger.info(f"Received WhatsApp message from {user_phone} to {to_number} (SID: {message_sid}): {message_text}")
        
        # Validate Twilio Account SID (optional security check)
        if TWILIO_ACCOUNT_SID and AccountSid != TWILIO_ACCOUNT_SID:
            logger.warning(f"Invalid Account SID: {AccountSid}")
            return _create_error_twiml("Authentication failed")
        
        # Validate message content
        if len(message_text) == 0:
            logger.warning("Empty message received")
            return _create_error_twiml("Please send a text message")
        
        if len(message_text) > 10000:  # Reasonable limit
            logger.warning(f"Message too long: {len(message_text)} chars")
            return _create_error_twiml("Message too long. Please send a shorter message.")
        
        # Process message through existing chatbot logic
        try:
            chatbot_response = process_chatbot_message(message_text, user_phone)
        except Exception as chatbot_error:
            logger.error(f"Chatbot processing error: {chatbot_error}")
            return _create_error_twiml("I'm having trouble processing your message. Please try again.")
        
        # Validate chatbot response
        if not chatbot_response or len(chatbot_response.strip()) == 0:
            logger.error("Empty chatbot response")
            return _create_error_twiml("I didn't understand that. Could you please rephrase?")
        
        # Create valid TwiML response
        twiml_response = _create_success_twiml(chatbot_response)
        
        logger.info(f"Sent TwiML response to {user_phone} (length: {len(chatbot_response)} chars)")
        
        return twiml_response
        
    except Exception as e:
        logger.error(f"Critical error in WhatsApp webhook: {e}")
        return _create_error_twiml("System error. Please try again later.")


def _create_success_twiml(message: str) -> str:
    """
    Create valid TwiML response for successful message delivery.
    """
    try:
        response = MessagingResponse()
        
        # Handle long messages by splitting if needed (WhatsApp limit is 4096 characters)
        if len(message) > 4000:
            # Split long messages into multiple parts
            parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    response.message(part)
                else:
                    # Add continuation indicator for subsequent parts
                    response.message(f"(continued) {part}")
        else:
            response.message(message)
        
        return str(response)
    except Exception as e:
        logger.error(f"Error creating TwiML: {e}")
        return _create_error_twiml("Response formatting error")


def _create_error_twiml(error_message: str) -> str:
    """
    Create valid TwiML response for error cases.
    """
    try:
        response = MessagingResponse()
        response.message(f"Error: {error_message}")
        return str(response)
    except Exception as e:
        logger.error(f"Critical error creating error TwiML: {e}")
        # Fallback TwiML
        return '<?xml version="1.0" encoding="UTF-8"?><Response><Message>System error occurred.</Message></Response>'


@app.post("/whatsapp/send-api")
async def send_whatsapp_via_api(request: Request):
    """
    Alternative endpoint that uses Twilio REST API to send messages.
    Useful for sending messages outside of webhook responses.
    """
    try:
        data = await request.json()
        
        to_number = data.get('to')
        message = data.get('message')
        
        if not to_number or not message:
            return {"error": "Missing 'to' or 'message' parameters"}, 400
        
        # Ensure phone number format
        if not to_number.startswith('whatsapp:+'):
            to_number = f"whatsapp:+{to_number.lstrip('+')}"
        
        success = send_whatsapp_message_via_api(to_number, message)
        
        if success:
            return {"status": "success", "message": "Message sent via REST API"}
        else:
            return {"error": "Failed to send message"}, 500
            
    except Exception as e:
        logger.error(f"Error in send-api endpoint: {e}")
        return {"error": "Internal server error"}, 500


@app.get("/whatsapp/webhook")
async def whatsapp_webhook_get():
    """
    Handle Twilio webhook verification and health check.
    """
    return {
        "status": "WhatsApp webhook is active",
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN),
        "whatsapp_number": TWILIO_WHATSAPP_NUMBER
    }


# ---------- Run server ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
