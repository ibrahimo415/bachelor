import os
import pandas as pd
import numpy as np

def compute_mos_from_votes(votes_1_to_10):
    votes = np.asarray(votes_1_to_10, dtype=np.float32)
    total = votes.sum()
    if total <= 0:
        return None
    scores = np.arange(1, 11, dtype=np.float32)  # 1..10
    return float((scores * votes).sum() / total)

def iter_samples(ava_root):
    """
    Your structure:
      ava_root/
        AVA_Files/AVA.txt
        images/<image_id>.jpg

    Yields:
      (image_id, image_path, mos)
    """
    ava_txt = os.path.join(ava_root, "AVA_Files", "AVA.txt")
    img_dir = os.path.join(ava_root, "images")

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
    for _, row in df.iterrows():
        image_id = str(int(row[1]))
        votes = row[2:12].values
        mos = compute_mos_from_votes(votes)
        if mos is None:
            continue

        image_path = os.path.join(img_dir, image_id + ".jpg")
        if not os.path.exists(image_path):
            continue

        yield image_id, image_path, mos
