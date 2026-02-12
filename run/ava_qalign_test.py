import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Pfad-Setup
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

# Scorer Import
from models.qalign_model import QAlignScorer, resize_max_side_lanczos

def main():
    img_root = Path(f"{BASE_STORAGE}/dataset/archive/images")
    gt_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_ground_truth.csv")
    out_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_TEST_1000.csv")

    print("📡 Lade Scorer...")
    scorer = QAlignScorer(device="cuda")

    # FIX: 'image_id' explizit als String laden, um .0 zu vermeiden
    df_test = pd.read_csv(gt_path, dtype={'image_id': str}).head(1000)

    results = []
    print(f"🚀 Starte Testlauf mit {len(df_test)} Bildern (ID-Fix aktiv)...")

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # image_id ist jetzt ein sauberer String
        img_id_str = row['image_id']
        img_path = img_root / f"{img_id_str}.jpg"

        if not img_path.exists():
            # Nur eine kleine Meldung, falls Bilder fehlen
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)

            scores = scorer.predict_from_pil(img_rsz)
            # Speichere die ID genau so, wie sie im String-Format vorliegt
            results.append({"image_id": img_id_str, "mos": row['mos'], **scores})
        except Exception as e:
            print(f"❌ Fehler bei {img_id_str}: {e}")

    # Speichern
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n✅ Test abgeschlossen! Datei unter: {out_path}")

if __name__ == "__main__":
    main()