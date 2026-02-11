import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# =========================================================
# 1. PFAD-SETUP & CACHE-UMLEITUNG
# =========================================================
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

BASE_STORAGE = "/data/stud/2026-BA-ibrahim_osman"

# WICHTIG: Cache-Umleitung VOR den Scorer-Imports
os.environ["HF_HOME"] = os.path.join(BASE_STORAGE, "huggingface_cache")
os.environ["TORCH_HOME"] = os.path.join(BASE_STORAGE, "torch_cache")
os.environ["TMPDIR"] = os.path.join(BASE_STORAGE, "torch_cache", "tmp")

from models.qalign_model import QAlignScorer, resize_max_side_lanczos

def get_config():
    if os.path.exists(BASE_STORAGE):
        return {
            "img_root": Path(f"{BASE_STORAGE}/dataset/PARA/imgs"),
            "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
            "out_dir": project_root / "results" / "features",
            "num_workers": 12,       # Erhöht für schnelles Image-Loading/Resize
            "prefetch_factor": 2,
            "pin_memory": True,
            "device": "cuda"
        }
    else:
        return {
            "img_root": Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs"),
            "gt_path": project_root / "results" / "features" / "para_ground_truth.csv",
            "out_dir": project_root / "results" / "features",
            "num_workers": 0,
            "device": "cpu"
        }

# =========================================================
# 2. DATASET KLASSE
# =========================================================
class PARADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row['session_id']) / str(row['original_name'])

        try:
            # CPU-Worker erledigen das Laden und Resizen
            img_pil = Image.open(img_path).convert("RGB")
            img_rsz, _ = resize_max_side_lanczos(img_pil, 1024)
            return img_rsz, row['image_id'], float(row['mos'])
        except Exception:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch[0] if batch else None

# =========================================================
# 3. MAIN RUN
# =========================================================
def main():
    config = get_config()
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "FULL"

    config["out_dir"].mkdir(parents=True, exist_ok=True)
    out_path = config["out_dir"] / f"para_QALIGN_SCORES_{mode}.csv"

    if not config["gt_path"].exists():
        print(f"❌ Fehler: Ground Truth CSV nicht gefunden: {config['gt_path']}")
        return

    df_full = pd.read_csv(config["gt_path"])

    # Split Logik für parallele Runs (A/B/FULL)
    mid = len(df_full) // 2
    if mode == "A": df_run = df_full.iloc[:mid]
    elif mode == "B": df_run = df_full.iloc[mid:]
    else: df_run = df_full

    print(f"📡 Lade Q-Align Scorer...")
    scorer = QAlignScorer(device=config['device'])

    dataset = PARADataset(df_run, config["img_root"])
    loader = DataLoader(
        dataset, batch_size=1, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"], collate_fn=collate_fn,
        prefetch_factor=config["prefetch_factor"] if config["num_workers"] > 0 else None
    )

    results = []
    print(f"🚀 Start Q-Align Run | Mode {mode} | {len(df_run)} Bilder")

    with torch.no_grad():
        for item in tqdm(loader, desc=f"PARA {mode}"):
            if item is None: continue
            img_rsz, img_id, mos = item

            try:
                # Inferenz
                scores = scorer.predict_from_pil(img_rsz)

                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores)
                results.append(entry)

                # Alle 500 Bilder zwischenspeichern
                if len(results) >= 500:
                    save_header = not out_path.exists()
                    pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)
                    results = []

            except Exception as e:
                print(f"\n⚠️ Fehler bei ID {img_id}: {e}")

    # Reste speichern
    if results:
        save_header = not out_path.exists()
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=save_header)

    print(f"\n✅ Fertig! Q-Align Ergebnisse unter: {out_path}")

if __name__ == "__main__":
    main()