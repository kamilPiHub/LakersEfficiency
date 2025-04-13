import pandas as pd

def min_max_normalize(df, features):
    return (df[features] - df[features].min()) / (df[features].max() - df[features].min())

def z_score_standardize(df, features):
    return (df[features] - df[features].mean()) / df[features].std()

def ratio_transformation(df, features):
    return df[features].div(df[features].mean())