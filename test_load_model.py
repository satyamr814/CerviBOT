# test_load_model.py
import os, sys, joblib, traceback

MODEL_CANDIDATES = [
    "xgb_cervical_pipeline.pkl",
    "model.pkl",
    "xgb_model.pkl",
]

print("Python:", sys.version.replace("\n"," "))
print("Working dir:", os.path.abspath(os.getcwd()))

found = False
for name in MODEL_CANDIDATES:
    path = os.path.join(os.getcwd(), name)
    print("Checking:", path, "exists?", os.path.exists(path))
    if os.path.exists(path):
        found = True
        print("Attempting to load:", path)
        try:
            m = joblib.load(path)
            print("Loaded object type:", type(m))
            # Quick checks
            print("Has predict:", hasattr(m, "predict"))
            print("Has predict_proba:", hasattr(m, "predict_proba"))
        except Exception as e:
            print("FAILED to load model. Traceback:")
            traceback.print_exc()
        break

if not found:
    print("No candidate model file found in current folder. Checked names:", MODEL_CANDIDATES)
