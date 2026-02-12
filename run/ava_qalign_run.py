import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# ✅ FIX 1: Tokenizer-Warnung unterdrücken (GANZ OBEN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pfad-Setup
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

# Cache-Umleitung
os.environ["HF_HOME"] = os.path.join(BASE_STORAGE, "huggingface_cache")
os.environ["TORCH_HOME"] = os.path.join(BASE_STORAGE, "torch_cache")

from models.qalign_model import QAlignScorer, resize_max_side_lanczos

class AVADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id_str = str(row['image_id'])
        img_path = self.img_root / f"{img_id_str}.jpg"

        if not img_path.exists():
            return None
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)
            return img_rsz, img_id_str, float(row['mos'])
        except:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch[0] if batch else None

def main():
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    img_root = Path(f"{BASE_STORAGE}/dataset/archive/images")
    gt_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_ground_truth.csv")
    out_path = Path(f"{BASE_STORAGE}/bachelor/results/features/ava_QALIGN_SCORES_{mode}_tmp.csv")

    df_full = pd.read_csv(gt_path, dtype={'image_id': str})

    n = len(df_full)
    if mode == "A": df_run = df_full.iloc[:n//3]
    elif mode == "B": df_run = df_full.iloc[n//3:2*n//3]
    else: df_run = df_full.iloc[2*n//3:]

    dataset = AVADataset(df_run, img_root)
    # ✅ FIX 2: num_workers=0 ist oft schneller auf GPU-Servern
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)

    # Scorer NACH DataLoader laden (beste Practice)
    print(f"📡 Lade Scorer...")
    scorer = QAlignScorer(device="cuda")

    results = []
    print(f"🚀 Start AVA {mode} | {len(df_run)} Bilder")

    for item in tqdm(loader, desc=f"AVA {mode}"):
        if item is None: continue
        img_rsz, img_id, mos = item
        try:
            scores = scorer.predict_from_pil(img_rsz)
            results.append({"image_id": img_id, "mos": mos, **scores})

            if len(results) >= 1000:
                pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
                results = []
        except:
            continue

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())

    print(f"✅ Fertig: {out_path}")

if __name__ == "__main__":
    main()