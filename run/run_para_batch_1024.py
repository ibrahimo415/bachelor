import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchmetrics.multimodal import CLIPImageQualityAssessment

# --- 1. DATASET (Windows-sicher ohne Lambda) ---
class PARADataset(Dataset):
    def __init__(self, img_paths, img_ids):
        self.img_paths = img_paths
        self.img_ids = img_ids
        self.transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert("RGB")
            img_tensor = self.transform(img)
            # WICHTIG: Multiplikation hier statt im Lambda (Windows Fix)
            img_tensor = img_tensor * 255.0
            return img_tensor, self.img_ids[idx]
        except:
            return None, self.img_ids[idx]

# --- 2. MAIN ---
def main():
    img_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA\imgs")
    out_path = Path(r"C:\Users\ibrah\bachelor\results\features\para_clip_1024_batch.csv")

    device = torch.device("cuda")

    # Bilder suchen
    all_imgs = list(img_root.glob("**/*.jpg"))
    img_ids = [f.stem for f in all_imgs]

    print(f"Gefundene Bilder: {len(all_imgs)}")

    dataset = PARADataset(all_imgs, img_ids)

    # num_workers=0 verhindert den Windows "Pickle/Spawn" Fehler komplett
    loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)

    # Alle 16 Prompts aus der Doku
    prompts = (
        "quality", "brightness", "noisiness", "colorfullness", "sharpness", "contrast",
        "complexity", "natural", "happy", "scary", "new", "warm", "real",
        "beautiful", "lonely", "relaxing"
    )

    # data_range=255.0 laut Doku einstellen
    metric = CLIPImageQualityAssessment(prompts=prompts, data_range=255.0).to(device)

    results = []
    print(f"🚀 Starte PARA-Turbo (1024px) auf {device}...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="PARA Verarbeitung"):
            imgs, ids = batch
            if imgs is None: continue

            imgs = imgs.to(device)
            scores = metric(imgs)

            for i in range(len(ids)):
                res = {"image_id": ids[i]}
                for p in prompts:
                    # Score aus dem Dict extrahieren und auf 4 Stellen runden
                    res[f"clip_{p}"] = round(float(scores[p][i].item()), 4)
                results.append(res)

            # Alle 1000 Bilder speichern (Flush)
            if len(results) >= 1000:
                pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())
                results = []

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', index=False, header=not out_path.exists())

    print(f"✅ Fertig! Ergebnisse in: {out_path}")

if __name__ == "__main__":
    main()