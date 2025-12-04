"""
Comprehensive training script for Cervical Cancer Risk Prediction Model
Handles data cleaning, preprocessing, and model training
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Expected feature order (must match app.py)
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
                # Fill with 'None' for categorical missing values
                df[col] = df[col].fillna('None')
                print(f"  Filled {missing_count} missing values in '{col}' with 'None'")
    
    # Handle numeric missing values (shouldn't be any, but just in case)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled missing values in '{col}' with median")
    
    # Convert target to binary
    print("\nProcessing target variable...")
    if 'Dx: Cancer' in df.columns:
        target_col = 'Dx: Cancer'
    else:
        # Try alternative names
        target_candidates = ['Dx:Cancer', 'Biopsy', 'Target']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("Could not find target column. Expected 'Dx: Cancer'")
    
    # Convert target to binary (Yes/No -> 1/0)
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
    feature_cols = []
    missing_features = []
    
    for feat in FEATURE_ORDER:
        if feat in df.columns:
            feature_cols.append(feat)
        else:
            missing_features.append(feat)
    
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
    
    X = df[feature_cols].copy()
    
    print(f"\nFinal dataset:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, feature_cols

def create_pipeline(X, feature_cols):
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
        from sklearn.preprocessing import OneHotEncoder
        preprocessor_steps.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
    
    if preprocessor_steps:
        preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='passthrough'
        )
    else:
        preprocessor = 'passthrough'
    
    # Create XGBoost model with optimized parameters
    # Note: use_label_encoder was removed in XGBoost 2.0+, so we don't include it
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
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
    pipeline = create_pipeline(X_train, feature_cols)
    
    # Train model
    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)
    print("\nThis may take a few minutes...")
    pipeline.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"\nTrain accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Predict probabilities for test set
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        print(f"\nTest predictions:")
        print(f"  Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
        print(f"  Mean probability: {y_proba.mean():.4f}")
    
    # Save model
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'backend', 'xgb_cervical_pipeline.pkl')
    
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
        print("Usage: python train_from_dataset.py <dataset.csv> [output_path]")
        print("\nExample:")
        print("  python train_from_dataset.py ../synth_3000_females_clean.csv")
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

