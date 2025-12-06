#!/bin/bash
# Test script for uploading model to the /upload-model endpoint

# Replace these with your actual values:
APP_URL="https://your-app.onrender.com"
MODEL_FILE="C:/path/to/xgb_cervical_pipeline.pkl"

# Corrected curl command:
# - Removed Content-Type header (curl sets it automatically with -F)
# - Use forward slashes or escaped backslashes for Windows paths
curl -X POST "${APP_URL}/upload-model" \
  -H "accept: application/json" \
  -F "file=@${MODEL_FILE}"

# Alternative for Windows PowerShell (use forward slashes):
# curl.exe -X POST "https://your-app.onrender.com/upload-model" `
#   -H "accept: application/json" `
#   -F "file=@C:/path/to/xgb_cervical_pipeline.pkl"

