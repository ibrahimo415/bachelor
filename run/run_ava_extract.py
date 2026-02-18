import os
import sys
from datetime import datetime
from pathlib import Path

# Sicherstellen, dass das Projekt-Hauptverzeichnis im Python-Pfad ist
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasets.ava import iter_samples
from extraction.extract_features import extract_dataset

def get_ava_root():
    """Erkennt automatisch die Umgebung und gibt den richtigen Pfad zurück."""
    # Pfad auf dem Uni-Server (megagpu)
    server_path = "/data/stud/2026-BA-ibrahim_osman/dataset/archive"

    if os.path.exists(server_path):
        return server_path

    # Pfade für lokale Entwicklung
    if os.name == "nt":  # Windows
        return r"C:\Users\ibrah\Desktop\dataset\archive"
    else:  # Mac
        return "/Users/ibrahim/Desktop/dataset/archive"

def main():
    ava_root = get_ava_root()

    # Zeitstempel für eindeutige Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ordner für Ergebnisse sicherstellen
    results_dir = os.path.join(base_dir, "results", "features")
    os.makedirs(results_dir, exist_ok=True)

    out_csv = os.path.join(results_dir, f"ava_features_{timestamp}.csv")

    print("--- Extraktions-Konfiguration ---")
    print(f"Umgebung: {'Uni-Server' if '/data/stud' in ava_root else 'Lokal'}")
    print(f"Datensatz-Pfad: {ava_root}")
    print(f"Ergebnis-Datei: {out_csv}")
    print("---------------------------------")

    extract_dataset(
        iter_samples(ava_root),
        out_csv,
        max_side=1024,
        limit=None,
        print_every=500,
        flush_every=500
    )

if __name__ == "__main__":
    main()