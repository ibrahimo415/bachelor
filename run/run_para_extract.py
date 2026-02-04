import os
import sys
from datetime import datetime

# Pfad-Fix (Zuerst!)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from datasets.para import iter_para_samples
from extraction.extract_features import extract_dataset

if __name__ == "__main__":
    # DEINE PFADE (jetzt präzise angepasst)
    para_root = r"/Users/ibrahim/Desktop/dataset/PARA"

    # Pfad zum annotation-Ordner
    metadata_files = [
        os.path.join(para_root, "annotation", "PARA-GiaaTrain.csv"),
        os.path.join(para_root, "annotation", "PARA-GiaaTest.csv")
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(base_dir, "results", "features", f"para_features_{timestamp}.csv")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"Starte PARA-Extraktion...")
    print(f"Suche Bilder in: {os.path.join(para_root, 'imgs')}")
    extract_dataset(
        iter_para_samples(para_root, metadata_files),
        out_csv,
        max_side=1024,
        limit=100,         #none #200
        print_every=50,   #5000 #50
        flush_every=500    #500 #50
    )