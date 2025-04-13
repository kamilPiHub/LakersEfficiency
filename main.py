from data.load_data import load_lakers_data
from preprocessing.classify_features import classify_features
from preprocessing.normalization import min_max_normalize, z_score_standardize, ratio_transformation
from preprocessing.outlier_detection import plot_boxplots
from ranking.synthetic_index import compute_synthetic_index
from visualization.plots import plot_histograms
import pandas as pd
import os

def main():
    filepath = "./data/Statistics.csv"
    df = load_lakers_data(filepath)

    features = classify_features()
    all_features = features['stimulants'] + features['destimulants']

    # Uzupełnienie braków danych medianą
    df[all_features] = df[all_features].fillna(df[all_features].median())

    # Uśrednienie cech przez liczbę lat gry
    for feature in all_features:
        df[feature] = df[feature] / df['Yrs']

    # Zamiana destymulant na stymulant (odwrotność)
    for feature in features['destimulants']:
        df[feature] = df[feature].max() - df[feature]

    # Wizualizacja cech oryginalnych
    plot_boxplots(df, all_features, output_dir="plots/boxplots")
    plot_histograms(df, all_features, output_dir="plots/histograms")

    # Zastosowanie trzech metod normalizacji i zapis rankingów
    transformations = {
        'unitaryzacja': min_max_normalize,
        'standaryzacja': z_score_standardize,
        'ilorazowa': ratio_transformation
    }

    for name, method in transformations.items():
        df_norm = method(df.copy(), all_features)
        df["Synthetic_Index"] = compute_synthetic_index(df_norm[all_features])
        ranking = df[["Player", "Synthetic_Index"]].sort_values(by="Synthetic_Index", ascending=False)
        output_file = f"ranking_lakers_{name}.csv"
        ranking.to_csv(output_file, index=False)
        print(f"TOP 5 ({name}):")
        print(ranking.head(5), "\n")

if __name__ == "__main__":
    main()
