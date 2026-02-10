import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# SETUP
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

from models.iqa_models import CLIPScorer

class AVADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Wir nutzen 'image_id' (wie per Terminal-Check bestätigt)
        img_id = str(int(row['image_id']))
        img_name = img_id + ".jpg"
        img_path = self.img_root / img_name
        return str(img_path), img_id

def collate_single(batch):
    return batch[0]

def main():
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    img_root = Path(r"C:\Users\ibrah\Desktop\dataset\archive\images")
    gt_path = project_root / "results" / "features" / "ava_ground_truth.csv"
    out_path = project_root / "results" / "features" / f"ava_CLIP_PART_{mode}.csv"

    # 1. VOLL-LAUF: Kein .head() mehr!
    df_full = pd.read_csv(gt_path)

    # Aufteilung in 3 Teile
    n = len(df_full)
    p = n // 3
    if mode == "A":
        df_run = df_full.iloc[:p]
    elif mode == "B":
        df_run = df_full.iloc[p:2*p]
    else:
        df_run = df_full.iloc[2*p:]

    scorer = CLIPScorer()
    dataset = AVADataset(df_run, img_root)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_single)

    results = []
    print(f"🚀 AVA VOLL-LAUF {mode} gestartet ({len(df_run)} Bilder)...")

    with torch.no_grad():
        for item in tqdm(loader, desc=f"AVA {mode}"):
            img_path, img_id = item
            if not os.path.exists(img_path):
                continue

            try:
                scores = scorer.predict(img_path)
                entry = {"image_id": img_id}
                entry.update(scores)
                results.append(entry)

                # 2. EFFIZIENZ: Alle 500 Bilder auf Festplatte schreiben
                if len(results) >= 500:
                    pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
                    results = []
            except Exception as e:
                print(f"Fehler bei ID {img_id}: {e}")

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
    print(f"✅ Teil {mode} komplett fertig!")

if __name__ == "__main__":
    main()