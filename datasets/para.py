import os
import pandas as pd
import numpy as np

def compute_para_mos_manual(row):
    scores = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    cols = [f"aestheticScore_{s:.1f}" for s in scores]
    votes = row[cols].values.astype(np.float32)
    total_votes = votes.sum()
    if total_votes <= 0: return None
    return float((scores * votes).sum() / total_votes)

def iter_para_samples(para_root, metadata_csv_list):
    # Der Ordner heißt laut deiner Info 'imgs'
    img_dir = os.path.join(para_root, "imgs")

    for csv_path in metadata_csv_list:
        if not os.path.exists(csv_path):
            print(f"CSV nicht gefunden: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            image_id = str(row['imageName'])
            session = str(row['sessionId'])

            mos = compute_para_mos_manual(row)
            if mos is None: continue

            # Pfad: .../PARA/imgs/session1/iaa_pub1_.jpg
            image_path = os.path.join(img_dir, session, image_id)

            if os.path.exists(image_path):
                yield image_id, image_path, mos
            # else: print(f"Bild fehlt: {image_path}") # Nur zum Testen einkommentieren