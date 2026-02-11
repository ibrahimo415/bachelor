import pandas as pd
from pathlib import Path
import os

if os.path.exists("/data/stud/2026-BA-ibrahim_osman/"):
    # --- SERVER (megagpu) ---
    para_root = Path("/data/stud/2026-BA-ibrahim_osman/dataset/PARA")
    output_dir = Path("/data/stud/2026-BA-ibrahim_osman/bachelor/results/features")
    print("🖥️  Server-Modus: Nutze /data/stud/... Pfade")
elif os.name == 'nt':
    # --- WINDOWS ---
    para_root = Path(r"C:\Users\ibrah\Desktop\dataset\PARA")
    output_dir = Path(r"C:\Users\ibrah\bachelor\results\features")
    print("💻 Windows-Modus erkannt")
else:
    # --- MAC ---
    para_root = Path("/Users/ibrahim/Desktop/dataset/PARA")
    output_dir = Path("/Users/ibrahim/Desktop/phase_01/results/features")
    print("🍎 Mac-Modus erkannt")

# Diese Variablen müssen aus den Pfaden oben zusammengebaut werden:
raw_csv = para_root / "annotation" / "PARA-Images.csv"
output_path = output_dir / "para_ground_truth.csv"

def prepare_gt():
    # Sicherheitscheck: Existiert die Quelldatei?
    if not raw_csv.exists():
        print(f"❌ FEHLER: Datei nicht gefunden: {raw_csv}")
        return

    print(f"Lade Rohdaten aus {raw_csv}...")
    df = pd.read_csv(raw_csv)

    print("Berechne MOS (Mittelwert) pro Bild...")
    # Gruppieren und Durchschnitt auf 4 Stellen runden
    gt = df.groupby(['imageName', 'sessionId'], sort=False)['aestheticScore'].mean().round(4).reset_index()

    print("Säubere IDs und sortiere Liste...")
    # 1. Spalten umbenennen
    gt.columns = ['original_name', 'session_id', 'mos']

    # 2. Saubere ID erstellen (iaa_pub1_.jpg -> iaa_pub1)
    gt['image_id'] = gt['original_name'].str.replace('.jpg', '', regex=False).str.rstrip('_')

    # 3. Natural Sort Vorbereitung (damit 1 vor 10 kommt)
    gt['temp_num'] = gt['image_id'].str.extract('(\d+)').astype(int)
    gt = gt.sort_values('temp_num').drop(columns=['temp_num'])

    # 4. Spalten anordnen
    gt = gt[['image_id', 'mos', 'session_id', 'original_name']]

    # Speichern
    output_dir.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output_path, index=False)
    print(f"\n✅ FERTIG! Ground Truth gespeichert unter:\n{output_path}")
    print(f"Anzahl Bilder: {len(gt)}")

if __name__ == "__main__":
    prepare_gt()