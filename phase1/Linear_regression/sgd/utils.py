import pandas as pd
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, target_column='FloodProbability'):
    df = df.drop(columns=['id'])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
