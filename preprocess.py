import pandas as pd

def load_data():
    df = pd.read_csv("data/dataset.csv")
    return df

def preprocess(df):
    X = df.drop("disease", axis=1)
    y = df["disease"]
    return X, y