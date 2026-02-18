import os
from PIL import Image
from models.musiq_model import MUSIQScorer

def main():
    # 1. Scorer initialisieren (nutzt deine neue Klasse)
    # CUDA_VISIBLE_DEVICES=3 im Terminal sorgt dafür, dass hier "cuda" reicht
    scorer = MUSIQScorer(model_type="musiq-ava", device="cuda")

    # Pfad zu einem deiner AVA Bilder
    image_path = "/data/stud/2026-BA-ibrahim_osman/dataset/archive/images/953777.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return

    # 2. Bild laden
    img_pil = Image.open(image_path).convert("RGB")

    # 3. Vorhersage treffen
    print(f"\n⏳ Berechne Score für {os.path.basename(image_path)}...")
    result = scorer.predict_from_pil(img_pil)

    # 4. Ergebnisse und Speicher anzeigen
    print("\n" + "="*40)
    print(f"Bild:  {os.path.basename(image_path)}")
    print(f"Score: {result['musiq_ava']:.4f}")
    print("="*40)

    # Hier ist die einmalige Speicher-Abfrage
    scorer.print_memory()

if __name__ == "__main__":
    main()