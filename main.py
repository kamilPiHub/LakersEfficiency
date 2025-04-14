import os
import pandas as pd
from data.load_data import load_lakers_data
from preprocessing.classify_features import classify_features
from preprocessing.methods import min_max_normalize, z_score_standardize, ratio_transformation
from visualization.plots import plot_histograms
from preprocessing.prepare_data import remove_outliers_isolation_forest, preprocess_data, transform_destimulants, generate_rankings

def main():
    filepath = "./data/DaneZawodnikow.csv"
    df = load_lakers_data(filepath)

    features = classify_features()
    all_features = features['stimulants'] + features['destimulants']

    df = remove_outliers_isolation_forest(df, all_features)
    df = preprocess_data(df, all_features)
    df = transform_destimulants(df, features['destimulants'])

    plot_histograms(df, all_features, output_dir="plots/histograms")

    transformations = {
        'unitaryzacja': min_max_normalize,
        'standaryzacja': z_score_standardize,
        'ilorazowa': ratio_transformation
    }

    generate_rankings(df, all_features, transformations)

if __name__ == "__main__":
    main()
