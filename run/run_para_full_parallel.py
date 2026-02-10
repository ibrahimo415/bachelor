import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 1. SETUP
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

from models.iqa_models import CLIPScorer

class PARADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row['session_id'] / row['original_name']
        return str(img_path), row['image_id'], float(row['mos'])

def collate_single(batch):
    return batch[0]

def main():
    # Modus A oder B
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "FULL"

    img_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs")
    gt_path = project_root / "results" / "features" / "para_ground_truth.csv"

    # Name der Datei für den Voll-Lauf
    out_path = project_root / "results" / "features" / f"para_FULL_POWER_{mode}.csv"

    # 1. ALLES LADEN (KEIN .head(100))
    df_full = pd.read_csv(gt_path)

    # 2. DYNAMISCHE AUFTEILUNG
    mid = len(df_full) // 2
    if mode == "A":
        df_run = df_full.iloc[:mid]
    elif mode == "B":
        df_run = df_full.iloc[mid:]
    else:
        df_run = df_full

    scorer = CLIPScorer()
    dataset = PARADataset(df_run, img_root)
    # Erhöht auf 4 Worker für schnellere SSD-Abfrage
    loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_single)

    results = []
    print(f"🚀 Voll-Lauf Prozess {mode} gestartet ({len(df_run)} Bilder)...")

    with torch.no_grad():
        for i, item in enumerate(tqdm(loader, desc=f"PARA {mode}")):
            img_path, img_id, mos = item
            if not os.path.exists(img_path): continue
            try:
                scores = scorer.predict(img_path)
                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores)
                results.append(entry)

                # 3. ZWISCHENSPEICHERN ALLE 500 BILDER
                if len(results) >= 500:
                    pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
                    results = []

            except Exception as e:
                print(f"\nFehler bei {img_id}: {e}")

    # Den letzten Rest speichern
    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())

    print(f"\n✅ Teil {mode} ist fertig! Datei: {out_path}")

if __name__ == "__main__":
    main()