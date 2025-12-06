"""
Preprocessing module for converting chatbot input to model-ready format
Handles Yes/No conversion, column ordering, and missing value handling
"""
import pandas as pd
import numpy as np
import os
import joblib
import logging

logger = logging.getLogger("cervi_backend")

# Expected feature order (must match training)
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


def yes_no_to_int(value):
    """
    Convert Yes/No strings to 0/1 integers.
    Handles various formats: 'Yes', 'No', 'yes', 'no', '1', '0', etc.
    """
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['yes', '1', 'true', 'positive']:
            return 1
        elif value_lower in ['no', '0', 'false', 'none', '', 'negative']:
            return 0
        else:
            # Try to convert directly if it's already a number string
            try:
                return int(float(value))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{value}' to int, defaulting to 0")
                return 0
    elif isinstance(value, (int, float)):
        return int(value)
    else:
        return 0


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert raw chatbot input dict into a one-row pandas DataFrame
    with the exact columns expected by the trained model.
    
    Automatically handles:
    - Yes/No string conversion to 0/1 for numeric fields
    - Column ordering
    - Missing value handling
    - Type conversion
    
    Args:
        data: Dictionary with user input. Keys can be in various formats:
            - Backend format: 'Hormonal_contraceptives' -> 'Yes'/'No'
            - Direct format: 'Hormonal contraceptives' -> 1/0
    
    Returns:
        pd.DataFrame: Single-row DataFrame ready for model prediction
    """
    # Map backend field names to model column names
    field_mapping = {
        'Age': 'Age',
        'Num_of_sexual_partners': 'Num of sexual partners',
        'First_sex_age': '1st sexual intercourse (age)',
        'Num_of_pregnancies': 'Num of pregnancies',
        'Smokes_years': 'Smokes (years)',
        'Hormonal_contraceptives': 'Hormonal contraceptives',
        'Hormonal_contraceptives_years': 'Hormonal contraceptives (years)',
        'STDs_HIV': 'STDs:HIV',
        'Pain_during_intercourse': 'Pain during intercourse',
        'Vaginal_discharge_type': 'Vaginal discharge (type- watery, bloody or thick)',
        'Vaginal_discharge_color': 'Vaginal discharge(color-pink, pale or bloody)',
        'Vaginal_bleeding_timing': 'Vaginal bleeding(time-b/w periods , After sex or after menopause)',
    }
    
    # Build row dictionary with proper column names
    row = {}
    
    for model_col in FEATURE_ORDER:
        # Try to find value in data dict
        value = None
        
        # First try direct column name match
        if model_col in data:
            value = data[model_col]
        # Then try field mapping
        else:
            for backend_key, mapped_col in field_mapping.items():
                if mapped_col == model_col and backend_key in data:
                    value = data[backend_key]
                    break
        
        # Convert value based on column type
        if value is None:
            # Handle missing values
            if model_col in ['Hormonal contraceptives', 'STDs:HIV']:
                value = 0  # Default to 0 for numeric Yes/No fields
            elif model_col in ['Pain during intercourse', 
                              'Vaginal discharge (type- watery, bloody or thick)',
                              'Vaginal discharge(color-pink, pale or bloody)',
                              'Vaginal bleeding(time-b/w periods , After sex or after menopause)']:
                value = 'None'  # Default for categorical
            else:
                value = 0  # Default for numeric
        else:
            # Convert Yes/No to 0/1 for numeric fields
            if model_col in ['Hormonal contraceptives', 'STDs:HIV']:
                value = yes_no_to_int(value)
            # Ensure numeric fields are numeric
            elif model_col in ['Age', 'Num of sexual partners', '1st sexual intercourse (age)',
                              'Num of pregnancies', 'Smokes (years)', 'Hormonal contraceptives (years)']:
                try:
                    value = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    value = 0.0
            # Ensure categorical fields are strings
            else:
                value = str(value) if value is not None else 'None'
        
        row[model_col] = value
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([row])
    df = df[[col for col in FEATURE_ORDER if col in df.columns]]
    
    logger.debug(f"Preprocessed input shape: {df.shape}")
    logger.debug(f"Preprocessed input columns: {list(df.columns)}")
    logger.debug(f"Preprocessed input dtypes:\n{df.dtypes}")
    
    return df


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate that required fields are present in the input data.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = [
        'Age', 'Num_of_sexual_partners', 'First_sex_age', 'Num_of_pregnancies'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, ""


if __name__ == "__main__":
    # Test preprocessing
    test_data = {
        'Age': 25,
        'Num_of_sexual_partners': 2,
        'First_sex_age': 18,
        'Num_of_pregnancies': 1,
        'Smokes_years': 0.0,
        'Hormonal_contraceptives': 'Yes',
        'Hormonal_contraceptives_years': 2.0,
        'STDs_HIV': 'No',
        'Pain_during_intercourse': 'No',
        'Vaginal_discharge_type': 'watery',
        'Vaginal_discharge_color': 'pale',
        'Vaginal_bleeding_timing': 'None',
    }
    
    df = preprocess_input(test_data)
    print("Test preprocessing:")
    print(df)
    print("\nData types:")
    print(df.dtypes)

