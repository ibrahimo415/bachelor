import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

try:
    from models.iqa_models import CLIPScorer
    print("CLIPScorer erfolgreich geladen.")
except ImportError:
    print("Fehler: Konnte 'models.iqa_models' nicht finden.")
    sys.exit()

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
    img_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs")
    gt_path = project_root / "results" / "features" / "para_ground_truth.csv"

    timestamp = datetime.now().strftime('%H-%M')
    out_path = project_root / "results" / "features" / f"para_TEST_safe_100_{timestamp}.csv"

    scorer = CLIPScorer()

    df_gt = pd.read_csv(gt_path).head(100)
    dataset = PARADataset(df_gt, img_root)

    loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_single)

    results = []
    print(f"Teste 100 Bilder mit CLIPScorer...")

    with torch.no_grad():
        for item in tqdm(loader, desc="Test-Lauf"):
            img_path, img_id, mos = item

            if not os.path.exists(img_path):
                continue

            try:
                scores_dict = scorer.predict(img_path)

                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores_dict)
                results.append(entry)
            except Exception as e:
                print(f"Fehler bei {img_id}: {e}")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nTest abgeschlossen!")
    print(f"Datei erstellt: {out_path}")

if __name__ == "__main__":
    main()