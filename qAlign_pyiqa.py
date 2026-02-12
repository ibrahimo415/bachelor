import os
from PIL import Image
from models.qalign_model import QAlignScorer, resize_max_side_lanczos

def main():
    scorer = QAlignScorer(device="cuda")  # NICHT cuda:3

  # /dataset/archive/images
    image_path = "/data/stud/2026-BA-ibrahim_osman/dataset/archive/images/953777.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return

    img_orig = Image.open(image_path).convert("RGB")
    img_rsz, _ = resize_max_side_lanczos(img_orig, 1024)

    print(f"\n🖼  Test: {os.path.basename(image_path)}")
    print("Original:", img_orig.size, "| Resized:", img_rsz.size)

    print("\n⏳ Scores (Original PIL):")
    res_orig = scorer.predict_from_pil(img_orig)

    print("⏳ Scores (1024 Resize PIL):")
    res_rsz = scorer.predict_from_pil(img_rsz)

    print("\n" + "="*65)
    print(f"{'Metrik':<20} | {'Original':<12} | {'1024px':<12} | {'Delta'}")
    print("-"*65)

    for k in res_orig.keys():
        a = res_orig[k]
        b = res_rsz[k]
        print(f"{k:<20} | {a:<12.4f} | {b:<12.4f} | {abs(a-b):.4f}")

    print("="*65)

if __name__ == "__main__":
    main()
