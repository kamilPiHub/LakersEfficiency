import numpy as np

def compute_synthetic_index(df, weights=None):
    """
    Calculates synthetic index using weighted sum method.
    """
    if weights is None:
        weights = np.ones(df.shape[1]) / df.shape[1]  # equal weights
    return df.dot(weights)