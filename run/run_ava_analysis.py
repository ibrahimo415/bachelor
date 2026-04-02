import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
sys.path.append(base_dir)

from analysis.plots import plot_low_mid_high

csv_filename = "ava_features_20260123_105605.csv"
csv_path = os.path.join(base_dir, "results", "features", csv_filename)

if __name__ == "__main__":
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        plot_output_dir = os.path.join(base_dir, "results", "plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        features_to_plot = ['brightness', 'contrast', 'colorfulness', 'sharpness', 'noise']

        for feat in features_to_plot:
            print(f"Erstelle Plot für: {feat}...")
            save_path = os.path.join(plot_output_dir, f"ava_{feat}_3bars.png")

            plot_low_mid_high(
                df,
                feat,
                title=f'Einfluss von {feat.capitalize()} (AVA)',
                save_path=save_path
            )

        print(f"\nFertig! Alle Grafiken sind in {plot_output_dir}")
    else:
        print(f"FEHLER: Datei nicht gefunden: {csv_path}")