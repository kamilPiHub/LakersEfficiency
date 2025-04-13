import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_histograms(df, features, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=20)
        plt.title(f'Histogram for {feature}')
        plt.savefig(f'{output_dir}/histogram_{feature}.png')
        plt.close()