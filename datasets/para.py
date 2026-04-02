import os
from pathlib import Path

import pandas as pd


def _normalize_image_id(image_name: str) -> str:
    # PARA-Dateinamen sind i.d.R. wie "iaa_pub1_.jpg"; GT nutzt "iaa_pub1"
    return Path(str(image_name)).stem.rstrip("_")


def _ensure_filename(image_name: str) -> str:
    image_name = str(image_name).strip()
    if not Path(image_name).suffix:
        return f"{image_name}.jpg"
    return image_name


def _iter_para_rows(df: pd.DataFrame):
    cols = set(df.columns)

    # Format A: bereits vorbereitetes GT-Format
    if {"image_id", "session_id", "original_name", "mos"}.issubset(cols):
        for _, row in df.iterrows():
            yield (
                str(row["image_id"]),
                str(row["session_id"]),
                _ensure_filename(row["original_name"]),
                float(row["mos"]),
            )
        return

    # Format B: PARA-Images.csv (Einzelratings)
    if {"imageName", "sessionId", "aestheticScore"}.issubset(cols):
        grouped = (
            df.groupby(["imageName", "sessionId"], sort=False)["aestheticScore"]
            .mean()
            .round(4)
            .reset_index(name="mos")
        )
        for _, row in grouped.iterrows():
            image_name = _ensure_filename(row["imageName"])
            yield (
                _normalize_image_id(image_name),
                str(row["sessionId"]),
                image_name,
                float(row["mos"]),
            )
        return

    # Format C: PARA-GiaaTrain/Test (aggregiertes Mittel)
    if {"imageName", "sessionId", "aestheticScore_mean"}.issubset(cols):
        for _, row in df.iterrows():
            image_name = _ensure_filename(row["imageName"])
            yield (
                _normalize_image_id(image_name),
                str(row["sessionId"]),
                image_name,
                float(row["aestheticScore_mean"]),
            )
        return

    raise ValueError(
        "Unsupported PARA metadata format. Expected one of: "
        "{image_id,session_id,original_name,mos} or "
        "{imageName,sessionId,aestheticScore} or "
        "{imageName,sessionId,aestheticScore_mean}."
    )


def iter_para_samples(para_root, metadata_csv_list):
    img_dir = Path(para_root) / "imgs"
    seen = set()

    for csv_path in metadata_csv_list:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        for image_id, session, original_name, mos in _iter_para_rows(df):
            if image_id in seen:
                continue
            image_path = img_dir / session / original_name
            if image_path.exists():
                seen.add(image_id)
                yield image_id, str(image_path), float(mos)
