# How to Upload Model to Your App

## Method 1: Using Python Script (Easiest) ✅

### For Local Server (http://127.0.0.1:8000)

```powershell
cd "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT"
python upload_model_test.py backend/xgb_cervical_pipeline.pkl
```

### For Render Deployment (https://your-app.onrender.com)

```powershell
cd "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT"
python upload_model_test.py backend/xgb_cervical_pipeline.pkl https://your-app.onrender.com
```

**Replace `https://your-app.onrender.com` with your actual Render URL!**

---

## Method 2: Using curl (Command Line)

### For Local Server

```powershell
curl -X POST "http://127.0.0.1:8000/upload-model" `
  -H "accept: application/json" `
  -F "file=@backend/xgb_cervical_pipeline.pkl"
```

### For Render Deployment

```powershell
curl -X POST "https://your-app.onrender.com/upload-model" `
  -H "accept: application/json" `
  -F "file=@backend/xgb_cervical_pipeline.pkl"
```

**Note:** Use forward slashes or escaped backslashes in the file path.

---

## Method 3: Using PowerShell Invoke-WebRequest

### For Local Server

```powershell
$filePath = "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT\backend\xgb_cervical_pipeline.pkl"
$uri = "http://127.0.0.1:8000/upload-model"

$form = @{
    file = Get-Item -Path $filePath
}

Invoke-WebRequest -Uri $uri -Method Post -Form $form
```

### For Render Deployment

```powershell
$filePath = "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT\backend\xgb_cervical_pipeline.pkl"
$uri = "https://your-app.onrender.com/upload-model"

$form = @{
    file = Get-Item -Path $filePath
}

Invoke-WebRequest -Uri $uri -Method Post -Form $form
```

---

## Method 4: Using Python requests (Direct)

```python
import requests

# For local server
url = "http://127.0.0.1:8000/upload-model"

# For Render
# url = "https://your-app.onrender.com/upload-model"

file_path = "backend/xgb_cervical_pipeline.pkl"

with open(file_path, 'rb') as f:
    files = {'file': (file_path, f, 'application/octet-stream')}
    response = requests.post(url, files=files)

print(response.json())
```

---

## Verify Upload Success

After uploading, check if the model is loaded:

```powershell
# Local
curl http://127.0.0.1:8000/health

# Render
curl https://your-app.onrender.com/health
```

You should see:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "..."
}
```

---

## Troubleshooting

### "Connection refused" or "Could not connect"
- **Local:** Make sure your server is running (`python app.py`)
- **Render:** Make sure your Render app is deployed and running

### "File not found"
- Make sure you're running the command from the `cerviBOT` directory
- Check the file path: `backend/xgb_cervical_pipeline.pkl`

### "Upload failed"
- Check the file size (should be ~0.5 MB)
- Make sure the file is a valid `.pkl` file
- Check server logs for detailed error messages

### For Render specifically:
- First deployment might take a few minutes
- Free tier services spin down after 15 min inactivity (they wake up automatically)
- Check Render dashboard for deployment status

---

## Quick Reference

**Model file location:**
```
C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT\backend\xgb_cervical_pipeline.pkl
```

**Relative path (from cerviBOT directory):**
```
backend/xgb_cervical_pipeline.pkl
```

**Recommended method:**
Use the Python script - it's the easiest and provides clear feedback!

```powershell
python upload_model_test.py backend/xgb_cervical_pipeline.pkl https://your-app.onrender.com
```

