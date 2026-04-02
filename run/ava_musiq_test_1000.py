import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

from models.musiq_model import MUSIQScorer

def resize_max_side_lanczos(img: Image.Image, max_side: int = 1024):
    """Konsistente Resize-Logik"""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS), True
    return img, False

def main():
    img_root = Path(f"{BASE_STORAGE}/dataset/archive/images")
    gt_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_ground_truth.csv")
    out_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_MUSIQ_TEST_1000.csv")

    print("Lade MUSIQ Scorer...")
    scorer = MUSIQScorer(model_type="musiq-ava", device="cuda")

    df_test = pd.read_csv(gt_path, dtype={'image_id': str}).head(1000)

    results = []
    print(f"Starte MUSIQ-Testlauf mit {len(df_test)} Bildern (1024px Resize)...")

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img_id_str = row['image_id']
        img_path = img_root / f"{img_id_str}.jpg"

        if not img_path.exists():
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")

            # Resizing für Konsistenz mit Q-Align/CLIP
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)

            scores = scorer.predict_from_pil(img_rsz)

            results.append({
                "image_id": img_id_str,
                "mos": row['mos'],
                **scores
            })
        except Exception as e:
            print(f"Fehler bei {img_id_str}: {e}")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nTest abgeschlossen! Datei erstellt unter: {out_path}")

    scorer.print_memory()

if __name__ == "__main__":
    main()