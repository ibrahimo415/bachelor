import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv

# Pfad-Setup
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
sys.path.append(str(base_dir))

from models.iqa_models import CLIPScorer

def main():
    # 1. PFADE & SETUP (Flexibel für Win/Mac)
    # Nutze Pathlib, das erkennt automatisch / oder \
    if os.name == 'nt':  # 'nt' bedeutet Windows
        img_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs")
        # Falls deine SSD einen anderen Buchstaben hat, hier anpassen
    else: # Mac/Linux
        img_root = Path("/Users/ibrahim/Desktop/dataset/PARA/imgs")

    gt_path = base_dir / "results" / "features" / "para_ground_truth.csv"

    if not gt_path.exists():
        print(f"Error: Ground Truth nicht gefunden unter {gt_path}")
        return

    df_gt = pd.read_csv(gt_path)
    scorer = CLIPScorer()

    # Ausgabedatei vorbereiten
    timestamp = datetime.now().strftime('%d.%m.%Y_%H-%M')
    out_name = f"para_clip_features_{timestamp}.csv"
    out_path = base_dir / "results" / "features" / out_name
    os.makedirs(out_path.parent, exist_ok=True)

    # 2. PARAMETER (Wie in Phase 01)
    limit = None
    print_every = 100
    flush_every = 500

    print(f"Starte CLIP-Extraktion...")
    print(f"Ziel: {out_path}")

    # 3. VERARBEITUNG MIT DIREKTEM SCHREIBEN
    with open(out_path, mode='w', newline='') as f:
        writer = None # Wird beim ersten Bild initialisiert

        for i, row in df_gt.iterrows():
            if limit and i >= limit: break

            img_id = row['image_id']
            img_path = img_root / row['session_id'] / row['original_name']

            if img_path.exists():
                try:
                    scores_dict = scorer.predict(str(img_path))

                    # Eintrag vorbereiten
                    entry = {"image_id": img_id, "mos": row['mos']}
                    entry.update(scores_dict)

                    # Header beim ersten erfolgreichen Bild schreiben
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=entry.keys())
                        writer.writeheader()

                    writer.writerow(entry)

                    # Fortschritt zeigen
                    if (i + 1) % print_every == 0:
                        print(f"[{i+1}/{len(df_gt)}] Verarbeitet: {img_id}")

                    # Auf Festplatte flashen
                    if (i + 1) % flush_every == 0:
                        f.flush()

                except Exception as e:
                    print(f"Fehler bei {img_id}: {e}")
            else:
                print(f" Bild fehlt: {img_path}")

    print(f"\n ERFOLG! Alle Daten sind sicher in:\n{out_path}")

if __name__ == "__main__":
    main()