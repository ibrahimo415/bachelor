import os
import sys
import csv
from datetime import datetime
from pathlib import Path

# Pfad-Setup
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
sys.path.append(str(base_dir))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from models.iqa_models import CLIPScorer
from datasets.ava import iter_ava_ids

def main():
    # AUTOMATISCHE PFAD-WEICHE
    if os.name == 'nt':  # Windows
        ava_root = Path(r"C:\Users\ibrah\Desktop\dataset\archive") # HIER Windows-Pfad prüfen
    else:                # Mac
        ava_root = Path("/Users/ibrahim/Desktop/dataset/archive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Der Rest nutzt "base_dir", das funktioniert auf beiden Systemen automatisch!
    out_path = base_dir / "results" / "features" / f"ava_features_only_{timestamp}.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    scorer = CLIPScorer()

    # --- HIER SIND DIE KONTROLL-VARIABLEN ---
    limit = None       # None für den echten Lauf
    print_every = 1000    # Info im Terminal nach jedem X-ten Bild
    flush_every = 2000 #100  # Sicherheits-Speicherung nach jedem X-ten Bild
    # ----------------------------------------

    print(f"Starte saubere Feature-Extraktion...")

    with open(out_path, mode='w', newline='') as f:
        writer = None

        for i, (img_id, img_path) in enumerate(iter_ava_ids(ava_root)):
            if limit and i >= limit:
                break

            try:
                scores_dict = scorer.predict(str(img_path))

                entry = {"image_id": img_id}
                entry.update(scores_dict)

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=entry.keys())
                    writer.writeheader()

                writer.writerow(entry)

                # JETZT MIT VARIABLE:
                if (i + 1) % flush_every == 0:
                    f.flush()

                if (i + 1) % print_every == 0:
                    print(f"[{i+1}] ID: {img_id} extrahiert.")

            except Exception as e:
                print(f" Fehler bei {img_id}: {e}")

    print(f"\n ERFOLG! Datei erstellt: {out_path}")

if __name__ == "__main__":
    main()