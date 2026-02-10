import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 1. SETUP: Projekt-Pfade (damit der Import von CLIPScorer klappt)
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

# Wir importieren DEINE Original-Klasse
try:
    from models.iqa_models import CLIPScorer
    print("✅ CLIPScorer erfolgreich aus deinem Projekt geladen.")
except ImportError:
    print("❌ Fehler: Konnte 'models.iqa_models' nicht finden. Prüfe deine Ordnerstruktur!")
    sys.exit()

# --- 2. DATASET ---
class PARADataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Pfadbau exakt wie in PARA gefordert
        img_path = self.img_root / row['session_id'] / row['original_name']
        return str(img_path), row['image_id'], float(row['mos'])

def collate_single(batch):
    return batch[0]

# --- 3. TEST-LAUF (100 Bilder) ---
def main():
    img_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs")
    gt_path = project_root / "results" / "features" / "para_ground_truth.csv"

    timestamp = datetime.now().strftime('%H-%M')
    out_path = project_root / "results" / "features" / f"para_TEST_safe_100_{timestamp}.csv"

    # Dein Original-Scorer
    scorer = CLIPScorer()

    # Nur die ersten 100 Bilder laden
    df_gt = pd.read_csv(gt_path).head(100)
    dataset = PARADataset(df_gt, img_root)

    # num_workers=4 lädt die Bilder schnell von der SSD vor
    loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=collate_single)

    results = []
    print(f"🧪 Teste 100 Bilder mit deinem Original-CLIPScorer...")

    with torch.no_grad():
        for item in tqdm(loader, desc="Test-Lauf"):
            img_path, img_id, mos = item

            if not os.path.exists(img_path):
                continue

            try:
                # Nutzt deine originale predict() Methode
                scores_dict = scorer.predict(img_path)

                entry = {"image_id": img_id, "mos": mos}
                entry.update(scores_dict)
                results.append(entry)
            except Exception as e:
                print(f"Fehler bei {img_id}: {e}")

    # Speichern der Test-CSV
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n✅ Test abgeschlossen!")
    print(f"📍 Datei erstellt: {out_path}")
    print(f"👉 Prüfe jetzt in IntelliJ, ob iaa_pub1 wieder bei 0.865 liegt.")

if __name__ == "__main__":
    main()