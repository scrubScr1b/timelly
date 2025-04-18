# utils.py
import os
import pandas as pd

UPLOAD_DIR = "uploaded_data"
CSV_PATH = os.path.join(UPLOAD_DIR, "dataset.csv")
EXCEL_PATH = os.path.join(UPLOAD_DIR, "dataset.xlsx")

def load_saved_dataset():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        source = "CSV"
    elif os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        source = "Excel"
    else:
        return None, None

    df.columns = df.columns.str.strip().str.lower()
    return df, source
