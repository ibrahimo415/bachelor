import os
import pandas as pd
import numpy as np

def compute_mos_from_votes(votes_1_to_10):
    votes = np.asarray(votes_1_to_10, dtype=np.float32)
    total = votes.sum()

    # Falls ein Bild keine Bewertungen hat, überspringen wir es
    if total <= 0:
        return None
    scores = np.arange(1, 11, dtype=np.float32)  # 1..10
    return float((scores * votes).sum() / total)

def iter_samples(ava_root):
    """
    Generator für Phase 01: Holt Bild-ID, Pfad und den berechneten MOS.
    Wichtig für die Korrelation von technischen Merkmalen mit der menschlichen Meinung.
    """

    # Pfade zu den Metadaten und dem Bilderordner
    ava_txt = os.path.join(ava_root, "AVA_Files", "AVA.txt")
    img_dir = os.path.join(ava_root, "images")

    # Prüfen ob alles da ist, damit das Skript nicht mitten im Lauf abstürzt
    if not os.path.exists(ava_txt):
        raise FileNotFoundError(f"AVA.txt not found: {ava_txt}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"images folder not found: {img_dir}")

    # AVA.txt has no header, whitespace-separated
    df = pd.read_csv(ava_txt, sep=r"\s+", header=None)

    # Format (from README):
    # col 0: index
    # col 1: image_id
    # col 2..11: vote counts for scores 1..10
    # Zeile für Zeile durchgehen
    for _, row in df.iterrows():
        image_id = str(int(row[1]))
        votes = row[2:12].values
        mos = compute_mos_from_votes(votes)
        if mos is None:
            continue

        image_path = os.path.join(img_dir, image_id + ".jpg")
        # Nur Bilder nehmen, die auch wirklich im Ordner existieren
        if not os.path.exists(image_path):
            continue

        yield image_id, image_path, mos

def iter_ava_ids(ava_root):
        """
        Liefert NUR ID und Pfad. Berechnet KEINEN MOS.
        Das spart Zeit und Rechenpower bei 250.000 Bildern.
        """
        ava_txt = os.path.join(ava_root, "AVA_Files", "AVA.txt")
        img_dir = os.path.join(ava_root, "images")

        if not os.path.exists(ava_txt):
            raise FileNotFoundError(f"AVA.txt nicht gefunden: {ava_txt}")

        # Wir laden nur Spalte 1 (die Image ID). Das geht super schnell.
        df = pd.read_csv(ava_txt, sep=r"\s+", header=None, usecols=[1])

        for _, row in df.iterrows():
            image_id = str(int(row[1]))
            image_path = os.path.join(img_dir, image_id + ".jpg")

            # Wir prüfen nur, ob das Bild da ist
            if os.path.exists(image_path):
                yield image_id, image_path