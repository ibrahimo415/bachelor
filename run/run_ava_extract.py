import os

from datasets.ava import iter_samples
from extraction.extract_features import extract_dataset
from datetime import datetime


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    if os.name == "nt":  # Windows
        ava_root = r"C:\Users\ibrah\Desktop\dataset\archive"
    else:  # Mac/Linux
        ava_root = r"/Users/ibrahim/Desktop/dataset/archive"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(base_dir, "results", "features", f"ava_features_{timestamp}.csv")

    print("Starte Extraktion...")
    print(f"Projekt-Hauptverzeichnis: {base_dir}")
    print(f"CSV wird gespeichert unter: {out_csv}")

    extract_dataset(
        iter_samples(ava_root),
        out_csv,
        max_side=1024,
        limit=None,         #none #200
        print_every=500,   #5000 #50
        flush_every=500     #500 #50
    )


if __name__ == "__main__":
    main()
