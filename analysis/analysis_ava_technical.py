import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================
# Daten laden
# ==========================
file_path = "results/features/ava_features_20260121_014102.csv"
df = pd.read_csv(file_path)

feature = 'sharpness'
target = 'mos'

data = df[[feature, target]].dropna().copy()

# ==========================
# Low / Mid / High (33%)
# ==========================
q_low = data[feature].quantile(0.33)
q_high = data[feature].quantile(0.66)

low = data[data[feature] <= q_low][target]
mid = data[(data[feature] > q_low) & (data[feature] < q_high)][target]
high = data[data[feature] >= q_high][target]

medians = [
    low.median(),
    mid.median(),
    high.median()
]

# ==========================
# Plot
# ==========================
plt.figure(figsize=(6, 4))
bars = plt.bar(['Low', 'Mid', 'High'], medians, color=['#d0d0d0', '#8fbcd4', '#2c7fb8'])

plt.ylabel('Median MOS')
plt.title('Einleitender Überblick: Schärfe vs. Ästhetik (AVA)')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Werte anzeigen
for bar, val in zip(bars, medians):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
