# Testing Guide for CerviBOT

## High-Risk Test Profile

Use this profile to test that the bot correctly identifies high-risk cases:

### JSON Format (for API testing):
```json
{
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
}
```

### Risk Factors in This Profile:
- **Older age** (45 years) - increases risk
- **Multiple sexual partners** (8) - significant risk factor
- **Early first sexual intercourse** (14 years) - risk factor
- **Multiple pregnancies** (5) - risk factor
- **Long-term smoking** (20 years) - risk factor
- **Long-term hormonal contraceptive use** (25 years) - risk factor
- **HIV positive** - major risk factor
- **Pain during intercourse** - symptom
- **Bloody vaginal discharge** - concerning symptom
- **Vaginal bleeding after sex** - concerning symptom

### Expected Result:
- **Risk Level**: HIGH
- **Expected Probability**: 60-90%
- **Advice**: "High risk — seek urgent clinical evaluation and further diagnostic testing."

## How to Test

### Method 1: Using cURL
```bash
curl -X POST https://your-render-app-url.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Method 2: Using the Frontend UI
1. Navigate to your deployed app URL
2. Enter the values from the high-risk profile above
3. Click "Get Risk Assessment"
4. Verify it shows HIGH risk with 60-90% probability

### Method 3: Using Python
```python
import requests

url = "https://your-render-app-url.onrender.com/predict"
data = {
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
}

response = requests.post(url, json=data)
print(response.json())
```

## Health Check Endpoint

Verify the model is loaded correctly:

```bash
curl https://your-render-app-url.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/model_files/cervical_cancer_model.pkl",
  "model_type": "CalibratedClassifierCV",
  "has_predict": true,
  "has_predict_proba": true,
  "test_prediction_works": true,
  "test_prediction_error": null,
  "version": "2.0.0",
  "checks_passed": true
}
```

## Model Loading Verification

The app now has **3 levels of model loading verification**:

1. **Import-time check**: Tries to load model when app.py is imported
2. **Startup event check**: Tries to load model on FastAPI startup
3. **Health endpoint check**: Actually tests that model can make predictions

All checks are logged, so you can see in the Render logs if the model loaded successfully.

## Render Deployment

The app is configured to automatically deploy from GitHub:
- **Repository**: https://github.com/satyamr814/CerviBOT.git
- **Branch**: main
- **Root Directory**: cerviBOT
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

After pushing to GitHub, Render will automatically:
1. Detect the push
2. Build the Docker container
3. Install dependencies
4. Copy model files
5. Start the application

## Troubleshooting

### If model predicts 0% for all cases:
The model needs to be retrained with the improved training script:
```bash
python backend/train_model.py <dataset.csv> cerviBOT/model_files/cervical_cancer_model.pkl
```

### If health check shows model not loaded:
1. Check Render logs for model loading errors
2. Verify `cerviBOT/model_files/cervical_cancer_model.pkl` exists in the repository
3. Check that the Dockerfile includes `COPY model_files/ ./model_files/`

### If test_prediction_works is false:
- Check the `test_prediction_error` field in the health response
- Verify the model file is not corrupted
- Ensure all dependencies are installed (especially imbalanced-learn)

