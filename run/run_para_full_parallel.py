import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

def get_config():
    if os.path.exists("/data/stud/2026-BA-ibrahim_osman/"):
        return {
            "img_root": Path("/data/stud/2026-BA-ibrahim_osman/dataset/PARA/imgs"),
            "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
            "out_dir": project_root / "results" / "features",
            "num_workers": 32,
            "prefetch_factor": 4,
            "pin_memory": True,
            "device": "cuda"
        }
    else:
        return {
            "img_root": Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs"),
            "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
            "out_dir": project_root / "results" / "features",
            "num_workers": 0,
            "prefetch_factor": None,
            "pin_memory": False,
            "device": "cpu"
        }

from models.iqa_models import CLIPScorer

class PARADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row['session_id']) / str(row['original_name'])
        return str(img_path), row['image_id'], float(row['mos'])

def collate_single(batch):
    return batch[0]

def main():
    config = get_config()
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "FULL"

    config["out_dir"].mkdir(parents=True, exist_ok=True)
    out_path = config["out_dir"] / f"para_FULL_POWER_{mode}.csv"

    if not config["gt_path"].exists():
        print(f"Fehler: CSV nicht gefunden unter {config['gt_path']}")
        return

    df_full = pd.read_csv(config["gt_path"])

    mid = len(df_full) // 2
    if mode == "A":
        df_run = df_full.iloc[:mid]
    elif mode == "B":
        df_run = df_full.iloc[mid:]
    else:
        df_run = df_full

    print(f"Initialisiere CLIPScorer auf {config['device']}...")
    scorer = CLIPScorer()

    dataset = PARADataset(df_run, config["img_root"])

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_single,
        prefetch_factor=config["prefetch_factor"] if config["num_workers"] > 0 else None
    )

    results = []
    print(f"Start: Mode {mode} | {len(df_run)} Bilder | Device: {config['device']}")
    print(f"Worker: {config['num_workers']} | Prefetch: {config['prefetch_factor']}")

    with torch.no_grad():
        for i, item in enumerate(tqdm(loader, desc=f"PARA {mode}")):
            img_path, img_id, mos = item

            if not os.path.exists(img_path):
                continue

            try:
                scores = scorer.predict(img_path)
                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores)
                results.append(entry)

                if len(results) >= 500:
                    save_header = not out_path.exists()
                    pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)
                    results = []

            except Exception as e:
                print(f"\nFehler bei Bild {img_id}: {e}")

    if results:
        save_header = not out_path.exists()
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)

    print(f"\nFertig. Ergebnisse unter: {out_path}")

if __name__ == "__main__":
    main()