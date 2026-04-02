import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

os.environ["HF_HOME"] = os.path.join(BASE_STORAGE, "huggingface_cache")
os.environ["TORCH_HOME"] = os.path.join(BASE_STORAGE, "torch_cache")
os.environ["TMPDIR"] = os.path.join(BASE_STORAGE, "torch_cache", "tmp")

from models.qalign_model import QAlignScorer, resize_max_side_lanczos

def get_config():
    return {
        "img_root": Path(f"{BASE_STORAGE}/dataset/archive/images"),
        "gt_path": Path(f"{BASE_STORAGE}/bachelor/results/features/ava_ground_truth.csv"),
        "out_dir": Path(f"{BASE_STORAGE}/bachelor/results/features"),
        "num_workers": 8, # 8 pro Prozess bei 4 Prozessen = 32 Worker insgesamt (perfekt!)
        "prefetch_factor": 2,
        "pin_memory": True,
        "device": "cuda"
    }

class AVADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / f"{int(row['image_id'])}.jpg"

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)
            return img_rsz, row['image_id'], float(row['mos'])
        except Exception:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch[0] if batch else None

def main():
    config = get_config()
    # Modi: A, B, C, D
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    config["out_dir"].mkdir(parents=True, exist_ok=True)
    out_path = config["out_dir"] / f"ava_QALIGN_SCORES_{mode}.csv"

    if not config["gt_path"].exists():
        print(f"Fehler: Ground Truth CSV nicht gefunden: {config['gt_path']}")
        return

    df_full = pd.read_csv(config["gt_path"])

    # Split Logik für 4 parallele Runs
    n = len(df_full)
    if mode == "A": df_run = df_full.iloc[:n//4]
    elif mode == "B": df_run = df_full.iloc[n//4:n//2]
    elif mode == "C": df_run = df_full.iloc[n//2:3*n//4]
    elif mode == "D": df_run = df_full.iloc[3*n//4:]
    else: df_run = df_full # Fallback auf alles

    print(f"Lade Q-Align Scorer auf {config['device']}...")
    scorer = QAlignScorer(device=config['device'])

    dataset = AVADataset(df_run, config["img_root"])
    loader = DataLoader(
        dataset, batch_size=1, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"], collate_fn=collate_fn,
        prefetch_factor=config["prefetch_factor"]
    )

    results = []
    print(f"Start AVA Q-Align | Mode {mode} | {len(df_run)} Bilder")

    with torch.no_grad():
        for item in tqdm(loader, desc=f"AVA {mode}"):
            if item is None: continue
            img_rsz, img_id, mos = item

            try:
                scores = scorer.predict_from_pil(img_rsz)
                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores)
                results.append(entry)

                # Alle 1000 Bilder zwischenspeichern (AVA ist groß!)
                if len(results) >= 1000:
                    save_header = not out_path.exists()
                    pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)
                    results = []

            except Exception as e:
                print(f"\nFehler bei ID {img_id}: {e}")

    if results:
        save_header = not out_path.exists()
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)

    print(f"\nFertig! Split {mode} gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()