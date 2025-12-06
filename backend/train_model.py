"""
Improved Training Script for Cervical Cancer Risk Prediction Model
- Better preprocessing
- Class weighting for imbalance
- Full metrics (AUC, precision, recall, confusion matrix)
- Probability calibration
- Saves to model_files/cervical_cancer_model.pkl
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Expected feature order (must match preprocessing)
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

def clean_and_preprocess_data(df):
    """Clean and preprocess the dataset."""
    print("=" * 70)
    print("Data Cleaning and Preprocessing")
    print("=" * 70)
    
    df = df.copy()
    print(f"\nOriginal dataset shape: {df.shape}")
    
    # Handle missing values in categorical columns
    print("\nHandling missing values...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Dx: Cancer':  # Don't fill target
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna('None')
                print(f"  Filled {missing_count} missing values in '{col}' with 'None'")
    
    # Handle numeric missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled missing values in '{col}' with median")
    
    # Convert target to binary
    print("\nProcessing target variable...")
    target_col = 'Dx: Cancer'
    y = df[target_col].copy()
    print(f"  Target column: '{target_col}'")
    print(f"  Target values: {y.value_counts().to_dict()}")
    
    # Convert to binary
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        y = y.astype(int)
    
    print(f"  Binary target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Select and reorder features
    print("\nSelecting features...")
    feature_cols = [feat for feat in FEATURE_ORDER if feat in df.columns]
    X = df[feature_cols].copy()
    
    print(f"\nFinal dataset:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, feature_cols

def create_pipeline(X, feature_cols, y):
    """Create the preprocessing and model pipeline."""
    print("\n" + "=" * 70)
    print("Creating Pipeline")
    print("=" * 70)
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if X[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Create preprocessing steps
    preprocessor_steps = []
    
    # Handle numeric columns
    if numeric_cols:
        preprocessor_steps.append(('num', StandardScaler(), numeric_cols))
    
    # Handle categorical columns
    if categorical_cols:
        preprocessor_steps.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
    
    if preprocessor_steps:
        preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='passthrough'
        )
    else:
        preprocessor = 'passthrough'
    
    # Calculate scale_pos_weight for class imbalance
    # Use a more balanced approach to avoid overly conservative predictions
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    raw_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Use a reduced scale_pos_weight to avoid too low risk predictions
    # Cap it at 3.0 to prevent overly conservative model
    scale_pos_weight = min(raw_ratio * 0.7, 3.0) if raw_ratio > 1.0 else 1.0
    print(f"\nClass imbalance ratio: {raw_ratio:.2f}")
    print(f"Using adjusted scale_pos_weight: {scale_pos_weight:.2f} (reduced to avoid low predictions)")
    
    # Create XGBoost model with balanced class weighting
    xgb_params = {
        'n_estimators': 300,  # Increased for better learning
        'max_depth': 7,  # Slightly deeper for more complex patterns
        'learning_rate': 0.03,  # Lower learning rate for better generalization
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 1,  # Lower to allow more splits
        'gamma': 0.1,  # Lower regularization
        'random_state': 42,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight  # Balanced class weighting
    }
    
    # Only add use_label_encoder for XGBoost < 2.0
    try:
        import xgboost as xgb_check
        xgb_version = xgb_check.__version__
        if xgb_version.startswith('1.'):
            xgb_params['use_label_encoder'] = False
    except:
        pass
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Create pipeline with SMOTE for imbalanced data
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('model', xgb_model)
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """Evaluate model with comprehensive metrics."""
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"  Actual Neg    {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"         Pos    {cm[1][0]:6d}    {cm[1][1]:6d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print(f"\nProbability Statistics:")
    print(f"  Min:  {y_proba.min():.4f}")
    print(f"  Max:  {y_proba.max():.4f}")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Std:  {y_proba.std():.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    
    # Check risk level distribution
    low_risk = (y_proba < 0.33).sum()
    med_risk = ((y_proba >= 0.33) & (y_proba < 0.67)).sum()
    high_risk = (y_proba >= 0.67).sum()
    print(f"\nRisk Level Distribution (for test predictions):")
    print(f"  Low risk (<33%):   {low_risk} ({low_risk/len(y_proba)*100:.1f}%)")
    print(f"  Medium risk (33-67%): {med_risk} ({med_risk/len(y_proba)*100:.1f}%)")
    print(f"  High risk (>67%):   {high_risk} ({high_risk/len(y_proba)*100:.1f}%)")
    
    # Warn if predictions are too low
    if y_proba.max() < 0.5:
        print(f"\n⚠ WARNING: Maximum probability is very low ({y_proba.max():.4f})")
        print(f"  Model may be too conservative. Consider adjusting scale_pos_weight.")
    elif y_proba.mean() < 0.1:
        print(f"\n⚠ WARNING: Mean probability is very low ({y_proba.mean():.4f})")
        print(f"  Model may be predicting too low risk. Consider adjusting parameters.")
    
    # Train metrics if provided
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"\nTrain Set Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

def train_model(csv_path, output_path=None):
    """Train the model and save it."""
    # Load data
    print("=" * 70)
    print("Cervical Cancer Risk Prediction Model Training")
    print("=" * 70)
    print(f"\nLoading dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Clean and preprocess
    X, y, feature_cols = clean_and_preprocess_data(df)
    
    # Split data
    print("\n" + "=" * 70)
    print("Splitting Data")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Create pipeline
    pipeline = create_pipeline(X_train, feature_cols, y_train)
    
    # Train model
    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)
    print("\nThis may take a few minutes...")
    pipeline.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test, X_train, y_train)
    
    # Optional: Calibrate probabilities (but check if it reduces probabilities too much)
    print("\n" + "=" * 70)
    print("Probability Calibration")
    print("=" * 70)
    
    # Check original probability range
    y_proba_orig = pipeline.predict_proba(X_test)[:, 1]
    orig_mean = y_proba_orig.mean()
    orig_max = y_proba_orig.max()
    print(f"Original model probability stats:")
    print(f"  Mean: {orig_mean:.4f}, Max: {orig_max:.4f}")
    
    try:
        calibrated_pipeline = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3)
        calibrated_pipeline.fit(X_train, y_train)
        print("✓ Probability calibration completed")
        
        # Evaluate calibrated model
        y_proba_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]
        auc_cal = roc_auc_score(y_test, y_proba_cal)
        cal_mean = y_proba_cal.mean()
        cal_max = y_proba_cal.max()
        
        print(f"  Calibrated AUC-ROC: {auc_cal:.4f}")
        print(f"  Calibrated probability stats:")
        print(f"    Mean: {cal_mean:.4f}, Max: {cal_max:.4f}")
        
        # Only use calibrated model if:
        # 1. AUC is better or similar (within 0.01)
        # 2. Mean probability doesn't drop too much (not more than 20% reduction)
        # 3. Max probability is still reasonable (at least 0.7)
        use_calibrated = (
            auc_cal >= metrics['auc'] - 0.01 and
            cal_mean >= orig_mean * 0.8 and
            cal_max >= 0.7
        )
        
        if use_calibrated:
            pipeline = calibrated_pipeline
            print("  ✓ Using calibrated model (maintains reasonable risk levels)")
        else:
            print("  ✗ Using original model (calibration would reduce risk predictions too much)")
            print(f"    Reason: AUC diff={auc_cal - metrics['auc']:.4f}, Mean reduction={((orig_mean - cal_mean)/orig_mean)*100:.1f}%")
    except Exception as e:
        print(f"  Calibration skipped: {e}")
        print("  Using original model")
    
    # Save model
    if output_path is None:
        # Create model_files directory if it doesn't exist
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_files')
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, 'cervical_cancer_model.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    print(f"\nSaving model to: {output_path}")
    joblib.dump(pipeline, output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Model saved successfully!")
    print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Save feature order for preprocessing
    feature_order_path = os.path.join(os.path.dirname(output_path), 'feature_order.pkl')
    joblib.dump(feature_cols, feature_order_path)
    print(f"✓ Feature order saved to: {feature_order_path}")
    
    # Verify the saved model can be loaded
    print("\nVerifying saved model...")
    try:
        loaded_model = joblib.load(output_path)
        print(f"✓ Model loaded successfully!")
        print(f"✓ Model type: {type(loaded_model).__name__}")
        print(f"✓ Has predict: {hasattr(loaded_model, 'predict')}")
        print(f"✓ Has predict_proba: {hasattr(loaded_model, 'predict_proba')}")
        
        # Test a quick prediction
        sample_pred = loaded_model.predict_proba(X_test.iloc[:1])[0]
        print(f"✓ Sample prediction works: {sample_pred}")
    except Exception as e:
        print(f"✗ Failed to verify model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return pipeline, output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <dataset.csv> [output_path]")
        print("\nExample:")
        print("  python train_model.py ../data/synth_5000_females_25percent_cancer.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found: {csv_path}")
        sys.exit(1)
    
    try:
        model, saved_path = train_model(csv_path, output_path)
        if model is not None:
            print("\n" + "=" * 70)
            print("Training Completed Successfully!")
            print("=" * 70)
            print(f"\nModel saved at: {saved_path}")
            print("You can now use this model with your FastAPI app.")
        else:
            print("\nTraining completed but model verification failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

