from .utils import load_data, clean_data, normalize_numeric, encode_categorical
import pandas as pd

def preprocess_dataset(file_path: str) -> pd.DataFrame:
    print("🔹 Loading dataset...")
    df = load_data(file_path)
    print(f"Dataset shape: {df.shape}")

    print("🔹 Cleaning data (remove duplicates, fill missing)...")
    df = clean_data(df)

    print("🔹 Encoding categorical features...")
    df = encode_categorical(df)

    print("🔹 Normalizing numeric features...")
    df = normalize_numeric(df)

    print("✅ Preprocessing complete.")
    return df

