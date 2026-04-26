import pandas as pd
import os, sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(BASE_DIR, "..", "..", "data", "medicines_1600_detailed.xlsx")

col_name       = "Medicine_Name"
col_components = "Component_List"
col_use        = "Used_For"
col_allergen   = "Allergen_Flags"
col_risk       = "Risk_Level"
col_type       = "Type"

def load_medicine_df():
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {os.path.abspath(file_path)}")
        sys.exit(1)

    df = pd.read_excel(file_path)
    print(f"[DB] Loaded {len(df)} medicines | columns: {list(df.columns)}")

    # Components separated by ' | '
    df[col_components] = df[col_components].fillna("").apply(
        lambda x: [i.strip().lower() for i in str(x).split("|") if i.strip()]
    )

    # Allergen flags also pipe-separated
    df[col_allergen] = df[col_allergen].fillna("").apply(
        lambda x: [i.strip().lower() for i in str(x).split("|") if i.strip()]
    )

    return df

medicine_df = load_medicine_df()
