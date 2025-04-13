import pandas as pd

def load_lakers_data(filepath):
    """
    Loads the Lakers player data from a CSV file with Windows-1250 encoding.
    """
    df = pd.read_csv(filepath, encoding='cp1250', sep=';')
    df = df.drop(columns=['Unnamed: 11'], errors='ignore')
    return df