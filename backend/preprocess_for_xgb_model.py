
import pandas as pd

EXPECTED_COLUMNS = [
    "Age",
    "Num of sexual partners",
    "1st sexual intercourse (age)",
    "Num of pregnancies",
    "Smokes (years)",
    "Hormonal contraceptives",
    "Hormonal contraceptives (years)",
    "STDs:HIV",
    "Pain during intercourse",
    "Vaginal discharge (type- watery, bloody or thick)",
    "Vaginal discharge(color-pink, pale or bloody)",
    "Vaginal bleeding(time-b/w periods , After sex or after menopause)",
]


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Turn raw chatbot input dict into a one-row pandas DataFrame
    with the exact columns expected by the trained model.

    The `data` dict should already contain values in the correct formats, e.g.:

      {
        "Age": 32,
        "Num of sexual partners": 4,
        "1st sexual intercourse (age)": 23,
        "Num of pregnancies": 3,
        "Smokes (years)": 2,
        "Hormonal contraceptives": 1,  # 1=yes, 0=no
        "Hormonal contraceptives (years)": 3,
        "STDs:HIV": 1,  # 1=positive, 0=negative
        "Pain during intercourse": "Yes",
        "Vaginal discharge (type- watery, bloody or thick)": "watery",
        "Vaginal discharge(color-pink, pale or bloody)": "pale",
        "Vaginal bleeding(time-b/w periods , After sex or after menopause)": "Between periods",
      }

    Any missing column will be filled with None.
    """
    row = {}
    for col in EXPECTED_COLUMNS:
        row[col] = data.get(col, None)
    df = pd.DataFrame([row])
    return df
