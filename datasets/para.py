import os
import pandas as pd
from pathlib import Path

def iter_para_samples(para_root, metadata_csv_list):
    img_dir = Path(para_root) / "imgs"

    for csv_path in metadata_csv_list:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Check: Wenn Einzelbewertungen vorliegen (PARA-Images.csv)
        if 'aestheticScore' in df.columns and 'imageName' in df.columns:
            print(f"Bilderliste erkannt. Berechne MOS aus Einzelbewertungen...")
            # Gruppieren nach Bild & Session, dann Durchschnitt der Noten berechnen
            # Mit sort=False bleibt die Reihenfolge der Original-Datei erhalten
            df = df.groupby(['imageName', 'sessionId'], sort=False)['aestheticScore'].mean().round(4).reset_index()
            # Spaltennamen für die Schleife vereinheitlichen
            df = df.rename(columns={'aestheticScore': 'mos', 'imageName': 'image_id'})

        for _, row in df.iterrows():
            img_id = row['image_id']
            session = row['sessionId']
            mos = row['mos']

            # Pfad: imgs/sessionX/iaa_pubX_.jpg
            image_path = img_dir / session / img_id

            if image_path.exists():
                yield img_id, str(image_path), float(mos)