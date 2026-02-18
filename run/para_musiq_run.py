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
from models.musiq_model import MUSIQScorer
from models.qalign_model import resize_max_side_lanczos

def get_config():
    return {
        "img_root": Path(f"{BASE_STORAGE}/dataset/PARA/imgs"),
        "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
        "out_path": project_root / "results" / "features" / "para_MUSIQ_FINAL_SCORES.csv",
        "device": "cuda",
    }

def main():
    cfg = get_config()

    print("📡 Initialisiere MUSIQ-AVA Scorer...")
    scorer = MUSIQScorer(model_type="musiq-ava", device=cfg["device"])

    if not cfg["gt_path"].exists():
        print(f"❌ Fehler: Ground Truth nicht gefunden: {cfg['gt_path']}")
        return

    # Wir laden die para_ground_truth.csv (die du auch für Q-Align genutzt hast)
    df_run = pd.read_csv(cfg["gt_path"])

    results = []
    print(f"🚀 Start PARA MUSIQ (Session-Logik) | {len(df_run)} Bilder")

    for _, row in tqdm(df_run.iterrows(), total=len(df_run), desc="PARA Processing"):
        # EXAKT DEINE PFAD-LOGIK: session_id / original_name
        img_path = cfg["img_root"] / str(row["session_id"]) / str(row["original_name"])

        if not img_path.exists():
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)

            # Prediction
            scores = scorer.predict_from_pil(img_rsz)

            # Wir speichern image_id und mos wie in deinem Q-Align Skript
            entry = {"image_id": row["image_id"], "mos": float(row["mos"])}
            entry.update(scores)
            results.append(entry)

            # Alle 500 Bilder speichern
            if len(results) >= 500:
                pd.DataFrame(results).to_csv(cfg["out_path"], mode='a', index=False, header=not cfg["out_path"].exists())
                results = []
        except Exception:
            continue

    if results:
        pd.DataFrame(results).to_csv(cfg["out_path"], mode='a', index=False, header=not cfg["out_path"].exists())

    print(f"\n✅ PARA abgeschlossen! Datei: {cfg['out_path']}")

if __name__ == "__main__":
    main()