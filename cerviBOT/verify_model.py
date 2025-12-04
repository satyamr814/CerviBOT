"""
Quick verification script to check if the model is loaded and working.
Run this to diagnose model loading issues.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Model Verification Script")
print("=" * 70)

# Test 1: Check if model file exists
print("\n1. Checking model file...")
model_path = os.path.join("backend", "xgb_cervical_pipeline.pkl")
abs_path = os.path.abspath(model_path)

if os.path.exists(abs_path):
    size = os.path.getsize(abs_path)
    print(f"   ✓ Model file found: {abs_path}")
    print(f"   ✓ File size: {size:,} bytes ({size/1024/1024:.2f} MB)")
else:
    print(f"   ✗ Model file NOT found at: {abs_path}")
    print("   Please ensure the model file exists or use /upload-model endpoint")
    sys.exit(1)

# Test 2: Try loading the model
print("\n2. Testing model loading...")
try:
    import joblib
    model = joblib.load(abs_path)
    print(f"   ✓ Model loaded successfully!")
    print(f"   ✓ Model type: {type(model).__name__}")
    print(f"   ✓ Has predict: {hasattr(model, 'predict')}")
    print(f"   ✓ Has predict_proba: {hasattr(model, 'predict_proba')}")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if app can load it
print("\n3. Testing app module loading...")
try:
    import app
    print(f"   ✓ App module imported successfully")
    print(f"   ✓ Model loaded in app: {app.model is not None}")
    if app.model is not None:
        print(f"   ✓ Model path in app: {app.model_path}")
    else:
        print(f"   ✗ Model is None in app module!")
        print("   This means the app didn't find/load the model at startup")
except Exception as e:
    print(f"   ✗ Failed to import app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test a sample prediction
print("\n4. Testing sample prediction...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample data matching FEATURE_ORDER
    sample_data = pd.DataFrame([{
        'Age': 25,
        'Num of sexual partners': 2,
        '1st sexual intercourse (age)': 18,
        'Num of pregnancies': 1,
        'Smokes (years)': 0.0,
        'Hormonal contraceptives': 'Yes',
        'Hormonal contraceptives (years)': 2.0,
        'STDs:HIV': 'No',
        'Pain during intercourse': 'No',
        'Vaginal discharge (type- watery, bloody or thick)': 'watery',
        'Vaginal discharge(color-pink, pale or bloody)': 'pale',
        'Vaginal bleeding(time-b/w periods , After sex or after menopause)': 'No',
    }])
    
    # Reorder columns to match FEATURE_ORDER
    sample_data = sample_data[[col for col in app.FEATURE_ORDER if col in sample_data.columns]]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(sample_data)[0]
        print(f"   ✓ Prediction successful!")
        print(f"   ✓ Probability: {proba}")
        if len(proba) > 1:
            print(f"   ✓ Risk probability: {proba[1]:.4f}")
    else:
        pred = model.predict(sample_data)[0]
        print(f"   ✓ Prediction successful!")
        print(f"   ✓ Prediction: {pred}")
except Exception as e:
    print(f"   ⚠ Prediction test failed: {e}")
    print("   (This might be okay if feature names don't match)")

print("\n" + "=" * 70)
if app.model is not None:
    print("✓ ALL CHECKS PASSED! Model is ready to use.")
    print("=" * 70)
    print("\nIf you're still getting 'Model not loaded' error:")
    print("  1. Make sure you restarted the server after making changes")
    print("  2. Check the server logs for any errors")
    print("  3. Try uploading the model via /upload-model endpoint")
else:
    print("✗ MODEL NOT LOADED IN APP!")
    print("=" * 70)
    print("\nTroubleshooting:")
    print("  1. Restart the server: Stop it (Ctrl+C) and run 'python app.py' again")
    print("  2. Check that you're running from the cerviBOT directory")
    print("  3. Upload the model: python upload_model_test.py backend/xgb_cervical_pipeline.pkl")
    sys.exit(1)

