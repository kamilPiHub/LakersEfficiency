from ranking.synthetic_index import compute_synthetic_index

def preprocess_data(df, all_features):
    df[all_features] = df[all_features].fillna(df[all_features].median())

    for feature in all_features:
        df[feature] = df[feature] / df['MP']

    return df

def transform_destimulants(df, destimulants):
    for feature in destimulants:
        df[feature] = df[feature].max() - df[feature]
    return df

def generate_rankings(df, all_features, method_dict):
    for name, method in method_dict.items():
        df_norm = method(df.copy(), all_features)
        df["Synthetic_Index"] = compute_synthetic_index(df_norm[all_features])
        ranking = df[["Player", "Synthetic_Index"]].sort_values(by="Synthetic_Index", ascending=False)
        output_file = f"ranking_lakers_{name}.csv"
        ranking.to_csv(output_file, index=False)
        print(f"TOP 5 ({name}):")
        print(ranking.head(5), "\n")