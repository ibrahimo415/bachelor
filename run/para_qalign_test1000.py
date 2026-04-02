import os
import sys
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

from models.qalign_model import QAlignScorer, resize_max_side_lanczos

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

def get_config():
    return {
        "img_root": Path(f"{BASE_STORAGE}/dataset/PARA/imgs"),
        "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
        "out_dir": project_root / "results" / "features",
        "num_workers": 4,
        "device": "cuda",
        "batch_size": 1,
        "flush_every": 100,
    }

class PARADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row["session_id"]) / str(row["original_name"])

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)
            return img_rsz, row["image_id"], float(row["mos"])
        except Exception:
            return None

def collate_fn(batch):
    # liefert immer eine LISTE (kann leer sein)
    return [x for x in batch if x is not None]

def main():
    cfg = get_config()
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    out_path = cfg["out_dir"] / f"para_QALIGN_TEST_1000_{mode}.csv"

    df_full = pd.read_csv(cfg["gt_path"])
    df_1000 = df_full.head(1000)

    df_run = df_1000.iloc[:500] if mode == "A" else df_1000.iloc[500:1000]
    df_run = df_run.reset_index(drop=True)

    print(f"Lade Q-Align fuer Teil {mode} ({len(df_run)} Bilder)...")
    scorer = QAlignScorer(device=cfg["device"])

    dataset = PARADataset(df_run, cfg["img_root"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=False,
    )

    results = []
    wrote_header = out_path.exists()

    t0 = time.time()
    processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Split {mode}"):
            if not batch:
                continue

            # batch ist Liste von (PIL, id, mos)
            for img_rsz, img_id, mos in batch:
                try:
                    scores = scorer.predict_from_pil(img_rsz)
                    entry = {"image_id": img_id, "mos": mos}
                    entry.update(scores)
                    results.append(entry)
                    processed += 1

                    # zwischenspeichern
                    if len(results) >= cfg["flush_every"]:
                        pd.DataFrame(results).to_csv(
                            out_path, mode="a", index=False, header=not wrote_header
                        )
                        wrote_header = True
                        results = []

                except Exception as e:
                    print(f"Fehler bei {img_id}: {e}")

    # rest schreiben
    if results:
        pd.DataFrame(results).to_csv(out_path, mode="a", index=False, header=not wrote_header)

    dt = time.time() - t0
    if processed > 0:
        print(f"Teil {mode} fertig! {processed} Bilder in {dt:.1f}s ({processed/dt:.2f} img/s)")
    else:
        print("Keine Bilder verarbeitet.")

if __name__ == "__main__":
    main()
