import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(df.mean(numeric_only=True))
    for col in df.columns:
        if df[col].isna().any() and df[col].dtype == 'object':
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna('unknown')
    return df

def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df
