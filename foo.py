import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor

# 1. SETUP PFADE & IMPORT SCORER
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))

from models.iqa_models import CLIPScorer

def run_comparison(image_path):
    scorer = CLIPScorer()
    img_path = Path(image_path)

    # --- A. ORIGINAL (HOCHAUFLÖSEND) ---
    img_orig = Image.open(img_path).convert("RGB")
    w_orig, h_orig = img_orig.size
    print(f"🚀 Starte Vergleich für: {img_path.name}")
    print(f"📏 Originalgröße: {w_orig} x {h_orig}")

    # Vorhersage für Original
    scores_orig = scorer.predict(str(img_path))

    # --- B. SKALIERT (1024px) ---
    # Wir machen das Gleiche wie im Hauptskript
    img_low = img_orig.copy()
    img_low.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    w_low, h_low = img_low.size
    print(f"📏 Skalierte Größe: {w_low} x {h_low}")

    # Da scorer.predict normalerweise einen Pfad will, nutzen wir hier einen
    # kleinen Trick und füttern den Tensor direkt (oder speichern kurz zwischen)
    # Für diesen Test speichern wir es kurz als Temp-Datei:
    temp_path = project_root / "results" / "tests" / "temp_lowres.jpg"
    img_low.save(temp_path, quality=95)
    scores_low = scorer.predict(str(temp_path))

    # --- C. VERGLEICH AUSGEBEN ---
    print("\n" + "="*50)
    print(f"{'Merkmal':<20} | {'Original':<10} | {'Skaliert':<10} | {'Diff'}")
    print("-" * 50)

    for key in scores_orig.keys():
        val_orig = scores_orig[key]
        val_low = scores_low[key]
        diff = abs(val_orig - val_low)
        # Wir markieren große Abweichungen (falls vorhanden)
        warning = "⚠️" if diff > 0.05 else "✅"
        print(f"{key:<20} | {val_orig:.4f}   | {val_low:.4f}   | {diff:.4f} {warning}")

    print("="*50)
    print("Hinweis: Abweichungen < 0.01 sind vernachlässigbar (Rundungsfehler/Resampling).")

if __name__ == "__main__":
    # Nimm ein richtig großes Bild für den Test
    sample_img = "/Users/ibrahim/Desktop/dataset/PARA/imgs/session2/iaa_pub110_.jpg"
    run_comparison(sample_img)