import os
import zipfile
from pathlib import Path

import pandas as pd


def _normalize_image_id(image_name: str) -> str:
    return Path(str(image_name).strip()).stem


def _ensure_filename(image_name: str) -> str:
    image_name = str(image_name).strip()
    if not Path(image_name).suffix:
        return f"{image_name}.jpg"
    return image_name


def _resolve_img_dir(tad_root: str) -> Path:
    root = Path(tad_root)
    candidates = [root, root / "images", root / "imgs", root / "TAD66K"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find TAD66K image folder in: {root}. "
        "Expected one of: root, root/images, root/imgs, root/TAD66K."
    )


def _iter_tad_rows(df: pd.DataFrame):
    cols = set(df.columns)

    if {"image", "score"}.issubset(cols):
        for _, row in df.iterrows():
            image_name = _ensure_filename(row["image"])
            yield _normalize_image_id(image_name), image_name, float(row["score"])
        return

    if {"image", "mos"}.issubset(cols):
        for _, row in df.iterrows():
            image_name = _ensure_filename(row["image"])
            yield _normalize_image_id(image_name), image_name, float(row["mos"])
        return

    if {"image_name", "mos"}.issubset(cols):
        for _, row in df.iterrows():
            image_name = _ensure_filename(row["image_name"])
            yield _normalize_image_id(image_name), image_name, float(row["mos"])
        return

    raise ValueError(
        "Unsupported TAD66K metadata format. Expected one of: "
        "{image,score}, {image,mos}, or {image_name,mos}."
    )


def iter_tad66k_samples(tad_root, metadata_csv_list):
    img_dir = _resolve_img_dir(tad_root)
    seen_ids = set()

    for csv_path in metadata_csv_list:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        for image_id, image_name, mos in _iter_tad_rows(df):
            if image_id in seen_ids:
                continue

            image_path = img_dir / image_name
            if image_path.exists():
                seen_ids.add(image_id)
                yield image_id, str(image_path), float(mos)


def iter_tad66k_samples_from_labels_zip(
    tad_root,
    labels_zip_path,
    csv_members=None,
):
    if csv_members is None:
        csv_members = ["labels/merge/train.csv", "labels/merge/test.csv"]

    img_dir = _resolve_img_dir(tad_root)
    seen_ids = set()

    with zipfile.ZipFile(labels_zip_path) as zf:
        available = set(zf.namelist())

        for member in csv_members:
            if member not in available:
                continue

            with zf.open(member) as f:
                df = pd.read_csv(f)

            for image_id, image_name, mos in _iter_tad_rows(df):
                if image_id in seen_ids:
                    continue

                image_path = img_dir / image_name
                if image_path.exists():
                    seen_ids.add(image_id)
                    yield image_id, str(image_path), float(mos)
