import os
from PIL import Image
from models.musiq_model import MUSIQScorer

def main():
    scorer = MUSIQScorer(model_type="musiq-ava", device="cuda")
    image_path = "/data/stud/2026-BA-ibrahim_osman/dataset/archive/images/953777.jpg"

    if not os.path.exists(image_path):
        print(f"Bild nicht gefunden: {image_path}")
        return

    img_pil = Image.open(image_path).convert("RGB")
    print(f"\nBerechne Score fuer {os.path.basename(image_path)}...")
    result = scorer.predict_from_pil(img_pil)

    print("\n" + "="*40)
    print(f"Bild:  {os.path.basename(image_path)}")
    print(f"Score: {result['musiq_ava']:.4f}")
    print("="*40)

    scorer.print_memory()

if __name__ == "__main__":
    main()