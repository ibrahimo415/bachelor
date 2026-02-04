import matplotlib.pyplot as plt
import os

def plot_low_mid_high(df, feature, target='mos', title=None, save_path=None):
    data = df[[feature, target]].dropna().copy()

    q_low = data[feature].quantile(0.33)
    q_high = data[feature].quantile(0.66)

    low = data[data[feature] <= q_low][target].median()
    mid = data[(data[feature] > q_low) & (data[feature] <= q_high)][target].median()
    high = data[data[feature] > q_high][target].median()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Low', 'Mid', 'High'], [low, mid, high], color=['#d0d0d0', '#8fbcd4', '#2c7fb8'], edgecolor='black')

    plt.ylabel(f'Median {target.upper()}')
    plt.title(title or f'{feature.capitalize()} vs. Ästhetik')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, val in zip(bars, [low, mid, high]):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close() # Wichtig, um Speicher zu sparen, wenn man viele Plots macht
    else:
        plt.show()