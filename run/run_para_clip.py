import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Pfad-Setup
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
sys.path.append(str(base_dir))

from models.iqa_models import CLIPScorer

def main():
    # Wir laden die soeben erstellte Ground Truth Datei
    gt_path = base_dir / "results" / "features" / "para_ground_truth.csv"
    img_root = Path("/Users/ibrahim/Desktop/dataset/PARA/imgs")

    if not gt_path.exists():
        print("❌ Error: Ground Truth Datei nicht gefunden! Bitte erst prepare_para_gt.py ausführen.")
        return

    df_gt = pd.read_csv(gt_path)
    scorer = CLIPScorer()
    results = []

    # LIMIT für Testlauf (auf None setzen für alle 31k Bilder)
    limit = 10

    print(f"Starte CLIP-Extraktion für {limit if limit else 'alle'} Bilder...")

    for i, row in df_gt.iterrows():
        if limit and i >= limit: break

        img_id = row['image_id']
        # Pfad zusammenbauen: imgs / sessionX / iaa_pubX_.jpg
        img_path = img_root / row['session_id'] / row['original_name']

        if img_path.exists():
            print(f"[{i+1}] Analysiere: {img_id}")
            scores_dict = scorer.predict(str(img_path))

            # Wir nehmen die Daten aus der GT-Datei und hängen die CLIP-Scores an
            entry = {"image_id": img_id, "mos": row['mos']}
            entry.update(scores_dict)
            results.append(entry)
        else:
            print(f"⚠️ Bild nicht gefunden: {img_path}")

    # Finales Speichern (Enthält jetzt ALLES: ID, MOS und CLIP-Scores)
    df_final = pd.DataFrame(results)
    out_name = f"para_full_features_clip_{datetime.now().strftime('%Y%m%d')}.csv"
    out_path = base_dir / "results" / "features" / out_name

    df_final.to_csv(out_path, index=False)
    print(f"\n✅ ERFOLG! Kombinierte Datei gespeichert:\n{out_path}")

if __name__ == "__main__":
    main()