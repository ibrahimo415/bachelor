import os
import sys
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from datasets.para import iter_para_samples
    from extraction.extract_features import extract_dataset
except ModuleNotFoundError as e:
    print(f"Fehler beim Importieren: {e}")
    print(f"Suche im Verzeichnis: {base_dir}")
    sys.exit(1)

if __name__ == "__main__":
    server_path = "/data/stud/2026-BA-ibrahim_osman/dataset/PARA"
    mac_path = "/Users/ibrahim/Desktop/dataset/PARA"

    if os.path.exists(server_path):
        para_root = server_path
    else:
        para_root = mac_path

    metadata_files = [
        os.path.join(para_root, "annotation", "PARA-GiaaTrain.csv"),
        os.path.join(para_root, "annotation", "PARA-GiaaTest.csv")
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(base_dir, "results", "features", f"para_features_{timestamp}.csv")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"--- PARA Konfiguration ---")
    print(f"Root-Pfad: {para_root}")
    print(f"Suche Bilder in: {os.path.join(para_root, 'imgs')}")
    print(f"Output-Datei: {out_csv}")
    print(f"--------------------------")

    extract_dataset(
        iter_para_samples(para_root, metadata_files),
        out_csv,
        max_side=1024,
        limit=None,
        print_every=100,  # Alle 100 Bilder ein Update
        flush_every=500
    )