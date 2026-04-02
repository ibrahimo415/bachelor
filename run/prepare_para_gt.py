import pandas as pd
from pathlib import Path
import os

if os.path.exists("/data/stud/2026-BA-ibrahim_osman/"):
    para_root = Path("/data/stud/2026-BA-ibrahim_osman/dataset/PARA")
    output_dir = Path("/data/stud/2026-BA-ibrahim_osman/bachelor/results/features")
    print("Server-Modus")
elif os.name == 'nt':
    para_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA")
    output_dir = Path(r"C:\Users\ibrah\bachelor\results\features")
    print("Windows-Modus")
else:
    para_root = Path("/Users/ibrahim/Desktop/dataset/PARA")
    output_dir = Path("/Users/ibrahim/Desktop/phase_01/results/features")
    print("Mac-Modus")

raw_csv = para_root / "annotation" / "PARA-Images.csv"
output_path = output_dir / "para_ground_truth.csv"

def prepare_gt():
    if not raw_csv.exists():
        print(f"Fehler: Datei nicht gefunden: {raw_csv}")
        return

    print(f"Lade Rohdaten aus {raw_csv}...")
    df = pd.read_csv(raw_csv)

    gt = df.groupby(['imageName', 'sessionId'], sort=False)['aestheticScore'].mean().round(4).reset_index()
    gt.columns = ['original_name', 'session_id', 'mos']
    gt['image_id'] = gt['original_name'].str.replace('.jpg', '', regex=False).str.rstrip('_')
    gt['temp_num'] = gt['image_id'].str.extract('(\d+)').astype(int)
    gt = gt.sort_values('temp_num').drop(columns=['temp_num'])
    gt = gt[['image_id', 'mos', 'session_id', 'original_name']]

    output_dir.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output_path, index=False)
    print(f"\nGround Truth gespeichert: {output_path}")
    print(f"Anzahl Bilder: {len(gt)}")

if __name__ == "__main__":
    prepare_gt()