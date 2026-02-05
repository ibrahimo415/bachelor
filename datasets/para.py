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

import os
import pandas as pd

def iter_para_samples(para_root, metadata_csv_list):
    img_dir = os.path.join(para_root, "imgs")

    for csv_path in metadata_csv_list:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # 1. MOS BERECHNEN (wie in Phase 02)
        # Wenn wir die PARA-Images.csv nutzen, müssen wir gruppieren
        if 'aestheticScore' in df.columns and 'imageName' in df.columns:
            print(f"Berechne MOS aus PARA-Images.csv...")
            df = df.groupby(['imageName', 'sessionId'], sort=False)['aestheticScore'].mean().round(4).reset_index()
            df = df.rename(columns={'aestheticScore': 'mos'})

        for _, row in df.iterrows():
            # 2. DATEINAME VS. ID (Der Trick mit dem Unterstrich)
            original_filename = str(row['imageName']) # Das ist "iaa_pub1_.jpg"
            session = str(row['sessionId'])

            # ID REINIGUNG: Hier machen wir es exakt wie in Phase 02
            # Entfernt .jpg und den Unterstrich am Ende
            image_id = original_filename.replace('.jpg', '').rstrip('_')

            mos = float(row['mos'])

            # Der Pfad zum Öffnen des Bildes braucht den ECHTEN Namen (mit _)
            image_path = os.path.join(img_dir, session, original_filename)

            if os.path.exists(image_path):
                # Wir geben die saubere image_id zurück für die CSV
                yield image_id, image_path, mos