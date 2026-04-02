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
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS), True
    return img, False

def main():
    img_root = Path(f"{BASE_STORAGE}/dataset/archive/images")
    gt_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_ground_truth.csv")
    out_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_MUSIQ_FINAL_SCORES.csv")

    print("Initialisiere MUSIQ-AVA Scorer...")
    scorer = MUSIQScorer(model_type="musiq-ava", device="cuda")

    df_run = pd.read_csv(gt_path, dtype={'image_id': str})

    results = []
    print(f"Start FULL RUN | {len(df_run)} Bilder")

    for _, row in tqdm(df_run.iterrows(), total=len(df_run), desc="MUSIQ Progress"):
        img_id_str = row['image_id']
        img_path = img_root / f"{img_id_str}.jpg"

        if not img_path.exists():
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            # 1024px Resize für Konsistenz zu Q-Align
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)

            scores = scorer.predict_from_pil(img_rsz)
            results.append({"image_id": img_id_str, "mos": row['mos'], **scores})

            # Alle 1000 Bilder speichern
            if len(results) >= 1000:
                pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
                results = []
        except Exception:
            continue

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())

    print(f"\nFertig! Datei: {out_path}")

if __name__ == "__main__":
    main()