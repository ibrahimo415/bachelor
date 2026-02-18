import csv
import os
import sys
from datetime import datetime
from pathlib import Path


script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datasets import iter_tad66k_samples
from models.iqa_models import CLIPScorer


def _get_tad_root() -> Path:
    if os.path.exists("/data/stud/2026-BA-ibrahim_osman/"):
        return Path("/data/stud/2026-BA-ibrahim_osman/dataset/TAD66K")
    if os.name == "nt":
        return Path(r"C:\Users\ibrah\Desktop\dataset\TAD66K")
    return Path("/Users/ibrahim/Desktop/dataset/TAD66K")


def _get_labels_csv() -> Path:
    candidates = [
        base_dir / "mos_tad66k" / "merge" / "all.csv",
        Path("/data/stud/2026-BA-ibrahim_osman/dataset/TAD66K/labels/merge/all.csv"),
        Path("/data/stud/2026-BA-ibrahim_osman/dataset/labels/merge/all.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "TAD66K labels CSV not found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}\n"
        f"- {candidates[2]}"
    )


def main(limit=None, print_every=500, flush_every=500):
    tad_root = _get_tad_root()
    labels_csv = _get_labels_csv()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / "results" / "features" / f"tad66k_clip_features_{timestamp}.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    scorer = CLIPScorer()

    print("--- TAD66K CLIP-IQA Konfiguration ---")
    print(f"Bilder: {tad_root}")
    print(f"Labels: {labels_csv}")
    print(f"Output: {out_path}")
    print("-------------------------------------")

    with open(out_path, mode="w", newline="") as f:
        writer = None

        for i, (img_id, img_path, mos) in enumerate(
            iter_tad66k_samples(tad_root=str(tad_root), metadata_csv_list=[str(labels_csv)])
        ):
            if limit and i >= limit:
                break

            try:
                scores_dict = scorer.predict(str(img_path))

                entry = {"image_id": img_id, "mos": float(mos)}
                entry.update(scores_dict)

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=entry.keys())
                    writer.writeheader()

                writer.writerow(entry)

                if (i + 1) % flush_every == 0:
                    f.flush()

                if (i + 1) % print_every == 0:
                    print(f"[{i + 1}] Verarbeitet: {img_id}")

            except Exception as e:
                print(f"Fehler bei {img_id}: {e}")

    print(f"\nFertig. Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()
