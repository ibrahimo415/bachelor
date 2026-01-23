import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.plots import plot_low_mid_high

# Den neuesten CSV-Pfad hier eintragen
csv_file = "results/features/ava_features_20260123_105605.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)

    # Beispiel Plot für Schärfe
    plt = plot_low_mid_high(df, 'sharpness', title='Schärfe vs. Ästhetik (AVA)')

    # Speichern im neuen Ordner
    os.makedirs("results/analysis", exist_ok=True)
    plt.savefig("results/analysis/ava_sharpness_3bars.png")
    plt.show()
else:
    print("CSV nicht gefunden!")