import os
import sys
from datetime import datetime
from pathlib import Path


script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasets import iter_tad66k_samples
from extraction.extract_features import extract_dataset


def _get_tad_root() -> Path:
    if os.path.exists("/data/stud/2026-BA-ibrahim_osman/"):
        return Path("/data/stud/2026-BA-ibrahim_osman/dataset/TAD66K")
    if os.name == "nt":
        return Path(r"C:\Users\ibrah\Desktop\dataset\TAD66K")
    return Path("/Users/ibrahim/Desktop/dataset/TAD66K")


if __name__ == "__main__":
    tad_root = _get_tad_root()
    labels_csv = Path(base_dir) / "mos_tad66k" / "merge" / "all.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(f"all.csv not found: {labels_csv}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(base_dir, "results", "features", f"tad66k_features_{timestamp}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"Starte TAD66K-Extraktion mit: {labels_csv}")
    print(f"Suche Bilder in: {tad_root}")
    print(f"Output CSV: {out_csv}")

    extract_dataset(
        iter_tad66k_samples(tad_root=str(tad_root), metadata_csv_list=[str(labels_csv)]),
        out_csv,
        max_side=1024,
        limit=None,
        print_every=500,
        flush_every=500,
    )
