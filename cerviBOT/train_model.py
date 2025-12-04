"""
Training script for Cervical Cancer Risk Prediction Model
Usage: python train_model.py <dataset.csv>
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

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Try to find the target column (common names)
    target_candidates = ['Biopsy', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Hinselmann', 'Schiller', 'Target']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use last column as target if no match found
        target_col = df.columns[-1]
        print(f"Warning: Using last column '{target_col}' as target. Please verify this is correct.")
    
    print(f"Using '{target_col}' as target column")
    
    # Select features (use FEATURE_ORDER if columns match)
    feature_cols = []
    for feat in FEATURE_ORDER:
        if feat in df.columns:
            feature_cols.append(feat)
        else:
            print(f"Warning: Feature '{feat}' not found in dataset")
    
    # If no features match, use all columns except target
    if len(feature_cols) == 0:
        print("No matching features found. Using all columns except target.")
        feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle target: convert to binary if needed
    if y.dtype == 'object' or y.nunique() > 2:
        print("Converting target to binary (0/1)")
        le = LabelEncoder()
        y = le.fit_transform(y)
        # If multiple classes, use the most common positive class
        if len(np.unique(y)) > 2:
            print(f"Warning: Multiple classes found: {np.unique(y)}. Using binary classification.")
            y = (y > 0).astype(int)
    else:
        y = y.astype(int)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, feature_cols

def create_pipeline(X, feature_cols):
    """Create the preprocessing and model pipeline."""
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if X[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Create preprocessing steps
    preprocessor_steps = []
    
    # Handle numeric columns
    if numeric_cols:
        preprocessor_steps.append(('num', StandardScaler(), numeric_cols))
    
    # Handle categorical columns (encode them)
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
    
    # Create XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Create pipeline with SMOTE for imbalanced data
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', xgb_model)
    ])
    
    return pipeline

def train_model(csv_path, output_path=None):
    """Train the model and save it."""
    # Load data
    X, y, feature_cols = load_and_prepare_data(csv_path)
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create pipeline
    print("\nCreating pipeline...")
    pipeline = create_pipeline(X_train, feature_cols)
    
    # Train model
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Predict probabilities for test set
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        print(f"Test predictions range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    
    # Save model
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'backend', 'xgb_cervical_pipeline.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving model to: {output_path}")
    joblib.dump(pipeline, output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"Model saved successfully! Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Verify the saved model can be loaded
    print("\nVerifying saved model...")
    loaded_model = joblib.load(output_path)
    print(f"✓ Model loaded successfully!")
    print(f"✓ Has predict: {hasattr(loaded_model, 'predict')}")
    print(f"✓ Has predict_proba: {hasattr(loaded_model, 'predict_proba')}")
    
    return pipeline, output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <dataset.csv> [output_path]")
        print("\nExample:")
        print("  python train_model.py data/cervical_cancer.csv")
        print("  python train_model.py data/cervical_cancer.csv backend/xgb_cervical_pipeline.pkl")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found: {csv_path}")
        sys.exit(1)
    
    try:
        model, saved_path = train_model(csv_path, output_path)
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"\nModel saved at: {saved_path}")
        print("You can now use this model with your FastAPI app.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

