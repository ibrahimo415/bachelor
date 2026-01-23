import os
import pandas as pd
from features.features import compute_features

def extract_dataset(samples, out_csv, max_side=1024, limit=None, print_every=5000, flush_every=500):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # --- Resume: IDs, die schon in der CSV sind, überspringen
    done_ids = set()
    if os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv, usecols=["image_id"])
            done_ids = set(prev["image_id"].astype(str).tolist())
            print(f"[RESUME] Found existing file with {len(done_ids)} rows. Skipping those IDs.")
        except Exception:
            print("[RESUME] Existing file exists but could not read 'image_id'. Will overwrite.")
            done_ids = set()
    current_custom_id = len(done_ids) + 1
    write_header = not os.path.exists(out_csv) or len(done_ids) == 0

    rows_buffer = []
    missing = 0
    processed_new = 0
    total_written = len(done_ids)

    for (image_id, image_path, mos) in samples:
        if limit is not None and processed_new >= limit:
            break

        image_id = str(image_id)
        if image_id in done_ids:
            continue

        feats = compute_features(image_path, max_side=max_side)
        if feats is None:
            missing += 1
            continue

        rows_buffer.append({
            "id": current_custom_id,
            "image_id": image_id,
            "mos": float(mos),
            **feats
        })
        current_custom_id += 1
        processed_new += 1

        # --- Batch schreiben
        if len(rows_buffer) >= flush_every:
            df = pd.DataFrame(rows_buffer)
            df.to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False
            total_written += len(df)
            rows_buffer.clear()

        if print_every and (processed_new % print_every == 0):
            print(f"[OK] new={processed_new} | total_written={total_written} | missing={missing}")

    # --- Rest schreiben
    if rows_buffer:
        df = pd.DataFrame(rows_buffer)
        df.to_csv(out_csv, mode="a", header=write_header, index=False)
        total_written += len(df)

    print(f"\nDone. New processed: {processed_new} | Total rows now: {total_written} | Missing: {missing}")
    print(f"Saved: {out_csv}")
