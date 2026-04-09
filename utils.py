import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df.ffill(inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df