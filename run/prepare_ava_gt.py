import pandas as pd
import numpy as np
from pathlib import Path

def compute_mos_from_votes(votes_1_to_10):
    votes = np.asarray(votes_1_to_10, dtype=np.float32)
    total = votes.sum()
    if total <= 0: return None
    scores = np.arange(1, 11, dtype=np.float32)
    mos_raw = (scores * votes).sum() / total
    return round(float(mos_raw), 4)

# --- SERVER PFADE ---
base_path = Path("/data/stud/2026-BA-ibrahim_osman")
ava_txt = base_path / "dataset/archive/AVA_Files/AVA.txt"
output_csv = base_path / "bachelor/results/features/ava_ground_truth.csv"

output_csv.parent.mkdir(parents=True, exist_ok=True)
print(f"⏳ Lese {ava_txt} ein...")

try:
    # AVA.txt hat kein Header und ist mit Leerzeichen getrennt
    df = pd.read_csv(ava_txt, sep=r"\s+", header=None)
    data = []

    for _, row in df.iterrows():
        # In AVA.txt ist Spalte 1 die ID (z.B. 953619)
        img_id = str(int(row[1]))
        # Spalten 2 bis 11 sind die Votes für Scores 1 bis 10
        votes = row[2:12].values
        mos = compute_mos_from_votes(votes)

        if mos is not None:
            data.append({"image_id": img_id, "mos": mos})

    final_df = pd.DataFrame(data)
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Erfolg! {len(final_df)} Bilder erfasst.")
    print(f"📍 Gespeichert unter: {output_csv}")

except Exception as e:
    print(f"❌ Fehler: {e}")