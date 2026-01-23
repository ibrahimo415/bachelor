import matplotlib.pyplot as plt
import pandas as pd

def plot_low_mid_high(df, feature, target='mos', title=None):
    data = df[[feature, target]].dropna().copy()

    q_low = data[feature].quantile(0.33)
    q_high = data[feature].quantile(0.66)

    low = data[data[feature] <= q_low][target].median()
    mid = data[(data[feature] > q_low) & (data[feature] <= q_high)][target].median()
    high = data[data[feature] > q_high][target].median()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Low', 'Mid', 'High'], [low, mid, high], color=['#d0d0d0', '#8fbcd4', '#2c7fb8'])

    plt.ylabel(f'Median {target}')
    plt.title(title or f'{feature.capitalize()} vs. {target}')

    for bar, val in zip(bars, [low, mid, high]):
        plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt