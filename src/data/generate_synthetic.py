"""
generate_synthetic.py
─────────────────────
Generates a synthetic CICIDS2017-like dataset for testing the pipeline
end-to-end without needing the 30GB raw dataset.

Produces a CSV file with all 78 feature columns, identity columns,
and a ` Label` column matching the CICIDS2017 schema.

Run:
  python src/data/generate_synthetic.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Insert path to access dummy features and label mapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.dummy import DEFAULT_FEATURE_COLUMNS
from src.data.label_mapper import UNIFIED_CLASSES

def generate_synthetic_data(num_rows: int = 20000, output_path: str = "data/raw/synthetic_cicids.csv"):
    np.random.seed(42)
    
    # 1. Identity Columns
    identity_cols = {
        "Flow ID": [f"192.168.1.1-10.0.0.{i%255}-{80}-{i%65535}-6" for i in range(num_rows)],
        "Source IP": ["192.168.1.1"] * num_rows,
        "Destination IP": [f"10.0.0.{i%255}" for i in range(num_rows)],
        "Source Port": [80] * num_rows,
        "Destination Port": [i % 65535 for i in range(num_rows)],
        "Protocol": [6] * num_rows,
        "Timestamp": pd.date_range("2017-07-04", periods=num_rows, freq="s").strftime("%d/%m/%Y %H:%M:%S")
    }
    
    df = pd.DataFrame(identity_cols)
    
    # 2. Feature Columns
    # Create random features, mostly gaussian noise
    for col in DEFAULT_FEATURE_COLUMNS:
        # standardise_columns in data_loader lowercase and replaces spaces with _. 
        # The default features in dummy.py are already formatted that way, but data_loader 
        # expects original CICIDS names. For our synthetic data, since data_loader will clean them,
        # providing them as-is (with underscores) is fine because data_loader strips and lowers.
        # But wait, data_loader expects CICIDS names. They don't strictly have to match the original
        # formatting, as long as after stripping and lowering they match `DEFAULT_FEATURE_COLUMNS`.
        df[col] = np.abs(np.random.normal(loc=50, scale=20, size=num_rows))
        
    # 3. Label Column
    # CICIDS2017 has " Label". The mapper uses UNIFIED_CLASSES but the raw data has original labels.
    # We can just provide the unified labels directly, as the mapper map_label(x) will map "Normal" to "Normal".
    
    # Distribution: 70% Normal, 30% Attacks
    probs = [0.70] + [0.30 / (len(UNIFIED_CLASSES) - 1)] * (len(UNIFIED_CLASSES) - 1)
    df[" Label"] = np.random.choice(UNIFIED_CLASSES, size=num_rows, p=probs)
    
    # Make sure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of synthetic data at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
