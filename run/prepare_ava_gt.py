import os
import pandas as pd
import numpy as np
from pathlib import Path

def compute_mos_from_votes(votes_1_to_10):
    votes = np.asarray(votes_1_to_10, dtype=np.float32)
    total = votes.sum()
    if total <= 0: return None

    scores = np.arange(1, 11, dtype=np.float32)
    # Hier passiert die Magie: Berechnung + Rundung auf 4 Stellen
    mos_raw = (scores * votes).sum() / total
    return round(float(mos_raw), 4)

# --- PFADE (jetzt mit "archive") ---
ava_root = Path(r"C:\Users\ibrah\Desktop\dataset\archive")
ava_txt = ava_root / "AVA_Files" / "AVA.txt"
output_csv = Path(r"C:\Users\ibrah\bachelor\results\features\ava_ground_truth.csv")

# Ordner erstellen falls er fehlt
output_csv.parent.mkdir(parents=True, exist_ok=True)

print("⏳ Erstelle AVA Ground Truth CSV (mit MOS-Rundung)...")

try:
    # AVA.txt einlesen
    df = pd.read_csv(ava_txt, sep=r"\s+", header=None)
    data = []

    for _, row in df.iterrows():
        img_id = str(int(row[1]))
        votes = row[2:12].values
        mos = compute_mos_from_votes(votes)

        if mos is not None:
            data.append({"image_id": img_id, "mos": mos})

    # Speichern
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Erfolg! {len(data)} Bilder erfasst.")
    print(f"📍 Datei: {output_csv}")

except FileNotFoundError:
    print(f"❌ Fehler: Die Datei wurde unter {ava_txt} nicht gefunden.")
    print("Prüfe bitte, ob der Ordner 'archive' wirklich auf dem Desktop liegt.")